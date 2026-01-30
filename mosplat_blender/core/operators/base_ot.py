from __future__ import annotations

from bpy.types import Context, WindowManager, Timer, Event, Operator

from typing import TYPE_CHECKING, Optional, Union, TypeVar, ClassVar, TypeAlias, Final
from typing import List, Set, Tuple, NamedTuple, Generic, Literal

import contextlib
from queue import Queue
import threading
from functools import partial

from ..handlers import load_dataset_property_group_from_json
from ..checks import check_addonpreferences, check_propertygroup, check_window_manager
from ...infrastructure.mixins import CtxPackage, MosplatContextAccessorMixin
from ...infrastructure.macros import immutable_to_set as im_to_set
from ...infrastructure.constants import (
    _TIMER_INTERVAL_,
    OPERATOR_ERROR_MAP,
    OPERATOR_FRAME,
)
from ...infrastructure.schemas import (
    UnexpectedError,
    DeveloperError,
    UserFacingError,
    OperatorIDEnum,
    MediaIODataset,
)
from ...interfaces.worker_interface import QT, MosplatWorkerInterface

if TYPE_CHECKING:
    from bpy.stub_internal.rna_enums import OperatorReturnItems as OpResult

    OpResultSet: TypeAlias = Set[OpResult]
    OpResultTuple: TypeAlias = Union[Tuple[OpResult, ...], OpResult]

    OpResultSetLike: TypeAlias = Union[OpResultTuple, OpResultSet]

    from ..preferences import Mosplat_AP_Global
    from ..properties import Mosplat_PG_Global

K = TypeVar("K", bound=NamedTuple)  # type of kwargs to async thread


class MosplatOperatorBase(Generic[QT, K], Operator, MosplatContextAccessorMixin):
    bl_category: ClassVar[str] = OperatorIDEnum._category()
    bl_options = {"REGISTER", "UNDO"}  # all of our operators should support undo

    __id_enum_type__ = OperatorIDEnum

    __worker: Optional[MosplatWorkerInterface[QT]] = None
    __timer: Optional[Timer] = None
    __data: Optional[MediaIODataset] = None

    _poll_error_msg_list: ClassVar[List[str]] = []  # can track all current poll errors

    @classmethod
    def at_registration(cls):
        super().at_registration()

        if cls.guard_type_of_bl_idname(cls.bl_idname, cls.__id_enum_type__):
            cls.bl_label = OperatorIDEnum.label_factory(cls.bl_idname)

    @classmethod
    def poll(cls, context) -> bool:
        cls._poll_error_msg_list.clear()

        try:
            check_addonpreferences(context.preferences)
        except (UserFacingError, UnexpectedError) as e:
            cls._poll_error_msg_list.append(str(e))
        try:
            check_propertygroup(context.scene)
        except (UserFacingError, UnexpectedError) as e:
            cls._poll_error_msg_list.append(str(e))
        try:
            check_window_manager(context.window_manager)
        except UserFacingError as e:
            cls._poll_error_msg_list.append(str(e))

        wrapped_result = cls._contexted_poll(cls.package(context))

        if len(cls._poll_error_msg_list) > 0:  # set the poll msg based on the list
            cls.poll_message_set("\n".join(cls._poll_error_msg_list))

        return wrapped_result

    @classmethod
    def _contexted_poll(cls, pkg: CtxPackage) -> bool:
        """an overrideable entrypoint for `poll` with access to prefs and props"""
        return True  # if not overriden will return true

    def invoke(self, context, event) -> OpResultSet:
        pkg = self.package(context)
        with self.CLEANUP_MANAGER(pkg):
            try:
                self.__load_global_data(pkg.prefs, pkg.props)

                wrapped_result: Final = im_to_set(self._contexted_invoke(pkg, event))
                # if not implemented run and return execute
                return wrapped_result
            except UserFacingError as e:  # all errs here are expected to be user-facing
                e.add_note("NOTE: Caught during operator invoke.")
                # only error as we have sufficiently covered the stack with messages
                self.error(str(e), "invoke")
                self.cleanup(pkg)
                return {"FINISHED"}  # return finished as blender data was modified

    def _contexted_invoke(self, pkg: CtxPackage, event: Event) -> OpResultSetLike:
        """
        an overrideable entrypoint for `execute` that ensures context is available as a property
        so that subsequently `props` and `prefs` properties can be accessed
        """
        return self.execute_with_package(pkg)  # if not overriden will just execute

    def execute(self, context) -> OpResultSet:
        return self.execute_with_package(self.package(context))

    def execute_with_package(self, pkg: CtxPackage) -> OpResultSet:
        with self.CLEANUP_MANAGER(pkg):
            try:
                wrapped_result: Final = im_to_set(self._contexted_execute(pkg))
            except AttributeError as e:
                msg = UserFacingError.make_msg(
                    "This error occured during operator execution.\n"
                    "Are you aware this operator requires invocation before execution?",
                    e,
                )
                self.error(msg, "execute")
                self.cleanup(pkg)
                return {"FINISHED"}  # finish because blender props changed

            if not ({"RUNNING_MODAL", "PASS_THROUGH"} & wrapped_result):  # intersection
                self.debug("Execution complete.")
                self.cleanup(pkg)  # cleanup if not a modal operator
            return wrapped_result

    def _contexted_execute(self, pkg: CtxPackage) -> OpResultTuple:
        """
        an overrideable entrypoint for `execute` that ensures context is available as a property
        so that subsequently `props` and `prefs` properties can be accessed
        """
        raise NotImplementedError  # this function is required

    def modal(self, context, event) -> OpResultSet:
        pkg = self.package(context)
        if event.type in {"ESC"}:
            self.info("Operator cancelled by user.")
            self.cleanup(pkg)
            return {"FINISHED"}  # user manually cancels through escape
        elif event.type != "TIMER":
            return {"PASS_THROUGH"}  # the event is not a timer callback
        with self.CLEANUP_MANAGER(pkg):
            wrapped_result: Final = im_to_set(self._contexted_modal(pkg, event))
            if not ({"RUNNING_MODAL", "PASS_THROUGH"} & wrapped_result) and self.worker:
                self.debug("Modal callbacks stopped.")
                # cleanup if a non-looping result was returned and worker is not None
                self.cleanup(pkg)
            return wrapped_result

    def _contexted_modal(self, pkg: CtxPackage, event: Event) -> OpResultTuple:
        """an overrideable entrypoint that abstracts away shared return paths in `modal` (see above)"""
        if not self.worker:
            return "FINISHED"

        if (next := self.worker.dequeue()) is not None:
            try:
                return self._queue_callback(pkg, event, next)
            finally:
                if pkg.context.area:
                    pkg.context.area.tag_redraw()  # redraw after queue callback

        return "RUNNING_MODAL"

    def _queue_callback(self, pkg: CtxPackage, event: Event, next: QT) -> OpResultTuple:
        """
        an entrypoint for when a new element is placed in the queue during `modal`.
        this function is required IF it's a modal operator.
        otherwise, the `NotImplementedError` pathway will never be seen.
        """
        raise NotImplementedError

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
            self.debug("Timer cleaned up")
        if self.worker:
            self.worker.cleanup()
            self.worker = None
            self.debug("Worker cleaned up")

        self.__data = None  # data is not guaranteed to be in-sync anymore
        self.info("Operator cleaned up")

    @contextlib.contextmanager
    def CLEANUP_MANAGER(self, pkg: CtxPackage):
        """ensures clean up always runs even with uncaught exceptions"""
        try:
            yield
        except BaseException as e:
            msg = DeveloperError.make_msg(
                "Uncaught exception during operator lifetime.", e
            )
            self.exception(msg)
            self.cleanup(pkg)  # cleanup here

    def launch_thread(self, context: Context, *, twargs: K):
        """`twargs` as in keyword args made of a immutable tuple"""

        worker_fn = partial(self._operator_thread, twargs=twargs)

        self.worker = MosplatWorkerInterface(self.bl_idname, worker_fn)
        self.worker.start()

        wm = self.wm(context)
        self.timer = wm.event_timer_add(
            time_step=_TIMER_INTERVAL_, window=context.window
        )
        wm.modal_handler_add(self)

    @staticmethod
    def _operator_thread(queue: Queue[QT], cancel_event: threading.Event, *, twargs: K):
        """
        this function is required IF it's a modal operator.
        otherwise, the `NotImplementedError` pathway will never be seen.
        """
        raise NotImplementedError

    """instance properties backed by mangled class attributes"""

    @property
    def worker(self) -> Optional[MosplatWorkerInterface[QT]]:
        return self.__worker

    @worker.setter
    def worker(self, wkr: Optional[MosplatWorkerInterface[QT]]):
        self.__worker = wkr

    @property
    def timer(self) -> Optional[Timer]:
        return self.__timer

    @timer.setter
    def timer(self, tmr: Optional[Timer]):
        self.__timer = tmr

    @property
    def data(self) -> MediaIODataset:
        """dataset property group as a dataclass."""
        if self.__data is None:
            raise DeveloperError("Dataset as dataclass not available in this scope.")
        else:
            return self.__data

    def wm(self, context: Context) -> WindowManager:
        if not (wm := context.window_manager):
            raise UnexpectedError("Poll-guard failed for window manager.")
        return wm

    """convenience functions to log and report"""

    def debug(self, msg: str):
        self.logger.debug(msg)
        self.report({"DEBUG"}, msg)

    def info(self, msg: str):
        self.logger.info(msg)
        self.report({"INFO"}, msg)

    def warn(self, msg: str):
        self.logger.warning(msg)
        self.report({"WARNING"}, msg)

    def error(self, msg: str, frame: OPERATOR_FRAME):
        self.logger.error(msg)
        self.report(OPERATOR_ERROR_MAP[frame], msg)

    def exception(self, msg: str):
        self.logger.exception(msg)
        self.report({"ERROR"}, msg)

    def __load_global_data(self, prefs: Mosplat_AP_Global, props: Mosplat_PG_Global):
        # try to restore both dataclass and property group from local JSON
        # mangle so that only base class can access
        try:
            self.__data, load_msg = load_dataset_property_group_from_json(props, prefs)
            self.info(load_msg)  # log the load message
        except UserWarning as e:  # raised if filesystem permission errors
            self.__data = None
            raise UserFacingError(
                "Operator cannot continue with insufficient permissions.", e
            ) from e
