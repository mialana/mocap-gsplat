from __future__ import annotations

import contextlib
import multiprocessing as mp
import multiprocessing.synchronize as mp_sync
import os
from dataclasses import dataclass
from functools import partial
from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
    Final,
    Generic,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
)

from infrastructure.constants import _TIMER_INTERVAL_
from infrastructure.identifiers import OperatorIDEnum
from infrastructure.macros import immutable_to_set as im_to_set
from infrastructure.mixins import ContextAccessorMixin, CtxPackage
from infrastructure.schemas import (
    DeveloperError,
    EnvVariableEnum,
    MediaIODataset,
    UnexpectedError,
    UserFacingError,
)
from interfaces.worker_interface import QT, MosplatWorkerInterface

if not EnvVariableEnum.SUBPROCESS_FLAG in os.environ or TYPE_CHECKING:
    from bpy.types import Operator, Timer  # pyright: ignore[reportRedeclaration]

    from core.checks import (
        check_addonpreferences,
        check_propertygroup,
        check_window_manager,
    )
    from core.handlers import load_dataset_property_group_from_json
else:
    Operator: TypeAlias = object
    Timer: TypeAlias = object
    check_addonpreferences = lambda c: c
    check_propertygroup = lambda c: c
    check_window_manager = lambda c: c
    load_dataset_property_group_from_json = lambda c: c


if TYPE_CHECKING:
    from bpy.stub_internal.rna_enums import OperatorReturnItems as OpResult
    from bpy.stub_internal.rna_enums import (
        OperatorTypeFlagItems,
    )
    from bpy.types import Context, Event, WindowManager

    OpResultSet: TypeAlias = Set[OpResult]
    OpResultTuple: TypeAlias = Union[Tuple[OpResult, ...], OpResult]

    OpResultSetLike: TypeAlias = Union[OpResultTuple, OpResultSet]

    from core.preferences import Mosplat_AP_Global
    from core.properties import Mosplat_PG_Global

OPERATOR_PROCESS_ENTRYPOINT_FN = "process_entrypoint"

K = TypeVar("K", bound=NamedTuple)  # type of kwargs to process


@dataclass
class MosplatOperatorMetadata:
    bl_idname: str
    bl_description: str
    bl_label: str
    bl_options: Set[OperatorTypeFlagItems]
    bl_category = OperatorIDEnum._category()

    def __init__(
        self,
        *,
        bl_idname: OperatorIDEnum,
        bl_description: str,
        bl_options: Set[OperatorTypeFlagItems] = {"REGISTER", "UNDO"},
    ):
        self.bl_idname = bl_idname.value
        self.bl_description = bl_description
        self.bl_label = OperatorIDEnum.label_factory(bl_idname)
        self.bl_options = bl_options


class MosplatOperatorBase(
    Generic[QT, K], ContextAccessorMixin[MosplatOperatorMetadata], Operator
):
    __worker: Optional[MosplatWorkerInterface[QT]] = None
    __timer: Optional[Timer] = None
    __data: Optional[MediaIODataset] = None

    _poll_error_msg_list: ClassVar[List[str]] = []  # can track all current poll errors

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
                e.add_note(f"NOTE: Caught error for '{self.bl_idname}' during invoke.")
                # only error as we have sufficiently covered the stack with messages
                self.logger.error(str(e))
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
                    f"This error occured during execution of '{self.bl_idname}'.\n"
                    "Are you aware this operator requires invocation before execution?",
                    e,
                )
                self.logger.error(msg)
                self.cleanup(pkg)
                return {"FINISHED"}  # finish because blender props have changed

            if not ({"RUNNING_MODAL", "PASS_THROUGH"} & wrapped_result):  # intersection
                self.logger.debug(f"'{self.bl_idname}' finished execution.")
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
            self.logger.info(f"'{self.bl_idname}' cancelled by user.")
            self.cleanup(pkg)
            return {"FINISHED"}  # user manually cancels through escape
        elif event.type != "TIMER":
            return {"PASS_THROUGH"}  # the event is not a timer callback
        with self.CLEANUP_MANAGER(pkg):
            wrapped_result: Final = im_to_set(self._contexted_modal(pkg, event))
            if not ({"RUNNING_MODAL", "PASS_THROUGH"} & wrapped_result) and self.worker:
                self.logger.debug(f"Modal callbacks stopped for '{self.bl_idname}'.")
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

    def _sync_to_props(self, props: Mosplat_PG_Global):
        """sync global properties with owned copy of data"""
        props.dataset_accessor.from_dataclass(self.data)

    def _queue_callback(self, pkg: CtxPackage, event: Event, next: QT) -> OpResultTuple:
        """
        an entrypoint for when a new element is placed in the queue during `modal`.
        this function is required IF it's a modal operator.
        otherwise, the `NotImplementedError` pathway will never be seen.
        """
        raise NotImplementedError

    def commit_data_to_json(self, pkg: CtxPackage):
        # update JSON with current state of PG as source of truth
        try:
            json_filepath = pkg.props.data_json_filepath(pkg.prefs)
            pkg.props.dataset_accessor.to_JSON(json_filepath)
            self.logger.info(
                f"Data from '{self.bl_idname}' committed to JSON: '{json_filepath}'"
            )
        except UserFacingError as e:
            msg = DeveloperError.make_msg(
                f"Error occurred while committing data from '{self.bl_idname}' back to JSON.",
                e,
            )
            self.logger.warning(msg)

    def cleanup(self, pkg: CtxPackage):
        self.commit_data_to_json(pkg)

        if self.timer:
            self.wm(pkg.context).event_timer_remove(self.timer)
            self.timer = None
            self.logger.debug("Timer cleaned up")
        if self.worker:
            self.worker.cleanup()
            self.worker = None
            self.logger.debug("Worker cleaned up")

        self.__data = None  # data is not guaranteed to be in-sync anymore
        self.logger.info(f"'{self.bl_idname}' cleaned up")

    def cancel(self, context):
        # no manager needed here as cleanup is non-blocking
        self.cleanup(self.package(context))

    @contextlib.contextmanager
    def CLEANUP_MANAGER(self, pkg: CtxPackage):
        """ensures clean up always runs even with uncaught exceptions"""

        def handle():
            msg = DeveloperError.make_msg(
                "Uncaught exception during operator lifetime.", e
            )
            self.logger.exception(msg)
            self.cleanup(pkg)  # cleanup here

        try:
            yield
        except Exception as e:
            handle()
        except BaseException as e:
            handle()
            raise  # this needs to be raised

    def launch_subprocess(self, context: Context, *, pwargs: K):
        """`pwargs` as in keyword args for subprocess"""

        try:
            process_entrypoint: Callable = self._operator_subprocess.__globals__[
                OPERATOR_PROCESS_ENTRYPOINT_FN
            ]
            assert isinstance(process_entrypoint, Callable)
        except (KeyError, AssertionError) as e:
            raise DeveloperError(
                f"'{self.bl_idname}' wants to use a process worker, "
                "but does not properly define a global process entrypoint function.",
                e,
            ) from e

        worker_fn = partial(process_entrypoint, pwargs=pwargs)

        self.worker = MosplatWorkerInterface(self.bl_idname, worker_fn)
        self.worker.start()

        wm = self.wm(context)
        self.timer = wm.event_timer_add(
            time_step=_TIMER_INTERVAL_, window=context.window
        )
        wm.modal_handler_add(self)

    @staticmethod
    def _operator_subprocess(
        queue: mp.Queue[QT], cancel_event: mp_sync.Event, *, pwargs: K
    ):
        """
        this function is required IF it's a modal operator.
        otherwise, this error is safe & correct as the pathway will never be seen.
        `pwargs` stands for both 'process kwargs'
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

    def __load_global_data(self, prefs: Mosplat_AP_Global, props: Mosplat_PG_Global):
        # try to restore both dataclass and property group from local JSON
        # mangle so that only base class can access
        try:
            self.__data, load_msg = load_dataset_property_group_from_json(props, prefs)
            self.logger.info(load_msg)  # log the load message
        except UserWarning as e:  # raised if filesystem permission errors
            self.__data = None
            raise UserFacingError(
                "Operator cannot continue with insufficient permissions.", e
            ) from e
