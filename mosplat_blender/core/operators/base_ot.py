from __future__ import annotations

from bpy.types import Context, WindowManager, Timer, Event, Operator

from typing import TYPE_CHECKING, Optional, Union, TypeVar, ClassVar, TypeAlias, Final
from typing import List, Set, Tuple, NamedTuple, Generic, FrozenSet

import contextlib
from queue import Queue
import threading

from ..checks import check_prefs_safe, check_props_safe
from ...infrastructure import mixins
from ...infrastructure.macros import (
    immutable_like_to_optional_set as oim_to_oset,
    immutable_to_set as im_to_set,
)
from ...infrastructure.decorators import worker_fn_auto
from ...infrastructure.schemas import (
    UnexpectedError,
    DeveloperError,
    UserFacingError,
    OperatorIDEnum,
    MediaIODataset,
)
from ...interfaces.worker_interface import MosplatWorkerInterface

if TYPE_CHECKING:
    from ..preferences import Mosplat_AP_Global
    from ..properties import Mosplat_PG_Global

    from bpy.stub_internal.rna_enums import OperatorReturnItems as OpResult

    OpResultSet: TypeAlias = Set[OpResult]
    OOpResultSet: TypeAlias = Optional[OpResultSet]
    OpResultTuple: TypeAlias = Union[Tuple[OpResult, ...], OpResult]
    OOpResultTuple: TypeAlias = Optional[OpResultTuple]

    OpResultSetLike: TypeAlias = Union[OpResultTuple, OpResultSet]

Q = TypeVar("Q")  # the type of the elements in worker queue, if used
K = TypeVar("K", bound=NamedTuple)


MODAL_FALLBACK_RESULT: Final[OpResultSet] = {"RUNNING_MODAL", "PASS_THROUGH"}


class CtxPackage(NamedTuple):
    context: Context
    prefs: Mosplat_AP_Global
    props: Mosplat_PG_Global


class MosplatOperatorBase(
    Generic[Q, K],
    Operator,
    mixins.MosplatBlTypeMixin,
    mixins.MosplatPGAccessorMixin,
    mixins.MosplatAPAccessorMixin,
):
    bl_category: ClassVar[str] = OperatorIDEnum._category()
    __id_enum_type__ = OperatorIDEnum
    _needs_invoke_before_execute: ClassVar[bool] = False

    __invoke_called = False
    __worker: Optional[MosplatWorkerInterface[Q]] = None
    __timer: Optional[Timer] = None
    __data: Optional[MediaIODataset] = None

    _poll_error_msg_list: ClassVar[List[str]] = []  # can track all current poll errors

    @classmethod
    def at_registration(cls):
        super().at_registration()

        if cls.guard_type_of_bl_idname(cls.bl_idname, cls.__id_enum_type__):
            cls.bl_label = OperatorIDEnum.label_factory(cls.bl_idname)

    @classmethod
    def package(cls, context: Context) -> CtxPackage:
        """convenience method to package context"""
        return CtxPackage(
            context=context, prefs=cls.prefs(context), props=cls.props(context)
        )

    @classmethod
    def poll(cls, context) -> bool:
        prefs = check_prefs_safe(context)
        props = check_props_safe(context)
        if prefs is None or props is None or context.window_manager is None:
            return False

        cls._poll_error_msg_list.clear()
        wrapped_result = cls.contexted_poll(cls.package(context))

        if len(cls._poll_error_msg_list) > 0:  # set the poll msg based on the list
            cls.poll_message_set("\n".join(cls._poll_error_msg_list))

        return wrapped_result or True  # if not implemented will return true

    @classmethod
    def contexted_poll(cls, pkg: CtxPackage) -> bool:
        """an overrideable entrypoint for `poll` with access to prefs and props"""
        ...

    """instance properties backed by mangled class attributes"""

    @property
    def worker(self) -> Optional[MosplatWorkerInterface[Q]]:
        return self.__worker

    @worker.setter
    def worker(self, wkr: Optional[MosplatWorkerInterface[Q]]):
        self.__worker = wkr

    @property
    def timer(self) -> Optional[Timer]:
        return self.__timer

    @timer.setter
    def timer(self, tmr: Optional[Timer]):
        self.__timer = tmr

    @property
    def data(self) -> MediaIODataset:
        """dataset property group as a dataclass"""
        if self.__data is None:
            raise DeveloperError(
                "Dataset as dataclass not available in this scope."
                "Correct usage is to call setter beforehand."
            )
        else:
            return self.__data

    @data.setter
    def data(self, mta: Optional[MediaIODataset]):
        self.__data = mta

    def wm(self, context: Context) -> WindowManager:
        if not (wm := context.window_manager):
            raise UnexpectedError("Poll-guard failed for window manager.")
        return wm

    @contextlib.contextmanager
    def CLEANUP_MANAGER(self, pkg: CtxPackage):
        """ensures clean up always runs even with uncaught exceptions"""
        try:
            yield
        except BaseException as e:
            dev_err = DeveloperError("Uncaught exception during operator lifetime.", e)
            self.logger.exception(str(dev_err))
            self.cleanup(pkg)  # cleanup here
            raise dev_err from e

    def invoke(self, context, event) -> OpResultSet:
        self.__invoke_called = True

        pkg = self.package(context)
        with self.CLEANUP_MANAGER(pkg):
            try:
                wrapped_result: Final = self.contexted_invoke(pkg, event)
                wrapped_parsed: Final = (
                    wrapped_result
                    if isinstance(wrapped_result, set)
                    else oim_to_oset(wrapped_result)
                )
                # if not implemented run and return execute
                return wrapped_parsed or self.execute(context)
            except UserFacingError as e:  # all errs here are expected to be user-facing
                e.add_note("NOTE: Caught during operator invoke.")
                self.logger.error(str(e))  # TODO: decide if needs to be `exception`

                self.cleanup(pkg)
                return {"CANCELLED"}

    def contexted_invoke(self, pkg: CtxPackage, event: Event) -> OpResultSetLike:
        """
        an overrideable entrypoint for `execute` that ensures context is available as a property
        so that subsequently `props` and `prefs` properties can be accessed
        """
        ...

    def execute(self, context, pkg: Optional[CtxPackage] = None) -> OpResultSet:
        pkg = pkg or self.package(context)

        if self.__class__._needs_invoke_before_execute and not self.__invoke_called:
            self.logger.error("This operator requires invokation.")
            self.cleanup(pkg)
            return {"CANCELLED"}

        props = pkg.props
        # set `data` property before execution
        self.data = props.dataset_accessor.to_dataclass()
        with self.CLEANUP_MANAGER(pkg):
            wrapped_result = im_to_set(self.contexted_execute(pkg))
            if not MODAL_FALLBACK_RESULT & wrapped_result:  # intersection
                self.cleanup(pkg)  # cleanup if not a modal operator
            return wrapped_result

    def contexted_execute(self, pkg: CtxPackage) -> OpResultTuple:
        """
        an overrideable entrypoint for `execute` that ensures context is available as a property
        so that subsequently `props` and `prefs` properties can be accessed
        """
        ...

    def modal(self, context, event) -> OpResultSet:
        pkg = self.package(context)
        if event.type in {"RIGHTMOUSE", "ESC"} or (
            self.worker is not None and self.worker.was_cancelled()
        ):
            self.cleanup(pkg)
            return {"CANCELLED"}
        elif event.type != "TIMER":
            return {"PASS_THROUGH"}
        with self.CLEANUP_MANAGER(pkg):
            wrapped_result = oim_to_oset(self.contexted_modal(pkg, event))
            if wrapped_result and not (MODAL_FALLBACK_RESULT & wrapped_result):
                # cleanup if a non-looping result was returned
                self.cleanup(pkg)  # cleanup before
            return wrapped_result or MODAL_FALLBACK_RESULT

    def contexted_modal(self, pkg: CtxPackage, event: Event) -> OOpResultTuple:
        """an overrideable entrypoint that abstracts away shared return paths in `modal` (see above)"""
        while self.worker is not None and (next := self.worker.dequeue()) is not None:
            try:
                return self.queue_callback(pkg, event, next)
            finally:
                if pkg.context.area:  # TODO: is redrawing spread out enough?
                    pkg.context.area.tag_redraw()  # redraw UI

    def queue_callback(self, pkg: CtxPackage, event: Event, next: Q) -> OOpResultTuple:
        """an entrypoint for when a new element is placed in the queue during `modal`"""
        ...

    def cancel(self, context):
        # no manager needed here as cleanup is non-blocking
        self.cleanup(self.package(context))

    def cleanup(self, pkg: CtxPackage):
        # update JSON with current state of PG as source of truth
        json_filepath = pkg.props.data_json_filepath(pkg.prefs)
        pkg.props.dataset_accessor.to_JSON(json_filepath)

        if self.timer:
            self.wm(pkg.context).event_timer_remove(self.timer)
            self.timer = None
            self.logger.debug("Timer cleaned up")
        if self.worker:
            self.worker.cleanup()
            self.worker = None
            self.logger.debug("Worker cleaned up")

        self.data = None  # data is not guaranteed to be in-sync anymore

    @staticmethod
    @worker_fn_auto
    def operator_thread(
        queue: Queue[Q],
        cancel_event: threading.Event,
        *,
        _kwargs: K,
    ): ...
