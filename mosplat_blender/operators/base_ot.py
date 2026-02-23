from __future__ import annotations

import contextlib
import importlib
import multiprocessing as mp
import multiprocessing.synchronize as mp_sync
import os
import queue
import sys
import threading
from dataclasses import dataclass
from functools import partial
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Final,
    Generator,
    Generic,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    cast,
)

from ..infrastructure.constants import _TIMER_INTERVAL_
from ..infrastructure.identifiers import OperatorIDEnum
from ..infrastructure.macros import immutable_to_set as im_to_set
from ..infrastructure.mixins import ContextAccessorMixin, CtxPackage
from ..infrastructure.protocols import SupportsToFromDict
from ..infrastructure.schemas import (
    AddonMeta,
    DeveloperError,
    EnvVariableEnum,
    MediaIOMetadata,
    UnexpectedError,
    UserFacingError,
)

# only used at runtime for generic typing
from ..interfaces.worker_interface import (
    QT,
    SubprocessWorkerInterface,
    ThreadWorkerInterface,
)

if not EnvVariableEnum.SUBPROCESS_FLAG in os.environ or TYPE_CHECKING:
    # ensure modules that use `bpy` are not imported in subprocess
    from bpy.types import Operator, Timer

    # completely avoid importing from core in subprocess
    from ..core.checks import check_window_manager
    from ..core.handlers import load_metadata_property_group_from_json
else:
    Operator: TypeAlias = object
    Timer: TypeAlias = object
    check_window_manager = lambda c: c
    load_metadata_property_group_from_json = lambda c: c

if TYPE_CHECKING:
    from bpy.stub_internal.rna_enums import (
        OperatorReturnItems as OpResult,
        OperatorTypeFlagItems,
    )
    from bpy.types import Context, Event, WindowManager

    OpResultSet: TypeAlias = Set[OpResult]
    OpResultTuple: TypeAlias = Union[Tuple[OpResult, ...], OpResult]

    OpResultSetLike: TypeAlias = Union[OpResultTuple, OpResultSet]

    from ..core.preferences import Mosplat_AP_Global
    from ..core.properties import Mosplat_PG_Global


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
    __subprocess_worker: Optional[SubprocessWorkerInterface[QT]] = None
    __thread_worker: Optional[ThreadWorkerInterface[QT]] = None
    __timer: Optional[Timer] = None
    __data: Optional[MediaIOMetadata] = None

    _poll_error_msg_list: ClassVar[List[str]] = []  # can track all current poll errors

    @classmethod
    def poll(cls, context) -> bool:
        cls._poll_error_msg_list.clear()

        try:
            check_window_manager(context.window_manager)
        except UserFacingError as e:
            cls._poll_error_msg_list.append(str(e))
        try:
            return cls._contexted_poll(cls.package(context))
        except (UserFacingError, UnexpectedError) as e:
            cls._poll_error_msg_list.append(str(e))
            return False
        finally:
            if len(cls._poll_error_msg_list) > 0:  # set the poll msg based on the list
                cls.poll_message_set("\n".join(cls._poll_error_msg_list))

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
            if not ({"RUNNING_MODAL", "PASS_THROUGH"} & wrapped_result) and (
                self.subprocess_worker or self.thread_worker
            ):
                self.logger.debug(f"Modal callbacks stopped for '{self.bl_idname}'.")
                # cleanup if a non-looping result was returned and worker is not None
                self.cleanup(pkg)
            return wrapped_result

    def _contexted_modal(self, pkg: CtxPackage, event: Event) -> OpResultTuple:
        """an overrideable entrypoint that abstracts away shared return paths in `modal` (see above)"""
        if not self.subprocess_worker and not self.thread_worker:
            return "FINISHED"

        next = (
            self.subprocess_worker.dequeue()
            if self.subprocess_worker
            else (self.thread_worker.dequeue() if self.thread_worker else None)
        )

        if next is not None:
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
        if (
            len(next) < 2
            or not isinstance(next[0], str)
            or not isinstance(next[1], str)
        ):
            raise DeveloperError("Queue type not compatible.")

        status = next[0]
        msg = next[1]
        match status:
            case "error":
                self.logger.error(msg)
                return "FINISHED"  # return finished as blender data has been modified
            case "warning":
                self.logger.warning(msg)
            case "done":
                self.logger.info(msg)
                return "FINISHED"  # return finished as blender data has been modified
            case "update":
                self.logger.debug(msg)
        return "RUNNING_MODAL"

    def cleanup(self, pkg: CtxPackage):
        self.commit_data_to_json(pkg)

        if self.timer:
            self.wm(pkg.context).event_timer_remove(self.timer)
            self.timer = None
            self.logger.debug("Timer cleaned up")
        self.cleanup_subprocess()
        self.cleanup_thread()

        self.__data = None  # data is not guaranteed to be in-sync anymore
        self.logger.info(f"'{self.bl_idname}' cleaned up")

    def cleanup_subprocess(self):
        if self.subprocess_worker:
            self.subprocess_worker.cleanup()
            self.logger.debug("Subprocess worker cleaned up")
            self.subprocess_worker = None

    def cleanup_thread(self):
        if self.thread_worker:
            self.thread_worker.cleanup()
            self.logger.debug("Thread worker cleaned up")
            self.thread_worker = None

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

    def sync_to_props(self, props: Mosplat_PG_Global):
        """sync global properties with owned copy of data"""
        props.media_io_accessor.from_dataclass(self.data)

    def commit_data_to_json(self, pkg: CtxPackage):
        # update JSON with current state of PG as source of truth
        try:
            json_filepath = pkg.props.media_io_metadata_filepath(pkg.prefs)
            pkg.props.media_io_accessor.to_JSON(json_filepath)
            self.logger.info(
                f"Data from '{self.bl_idname}' committed to JSON: '{json_filepath}'"
            )
        except UserFacingError as e:
            msg = DeveloperError.make_msg(
                f"Error occurred while committing data from '{self.bl_idname}' back to JSON.",
                e,
            )
            self.logger.warning(msg)

    def launch_subprocess(self, context: Context, *, pwargs: K):
        """`pwargs` as in keyword args for subprocess"""

        with self.SUBPROCESS_MANAGER(pwargs) as (
            process_entrypoint,
            ImportedWorkerInterface,
            new_pwargs,
        ):
            worker_fn = partial(process_entrypoint, pwargs=new_pwargs)

            self.subprocess_worker = ImportedWorkerInterface(self.bl_idname, worker_fn)
            self.subprocess_worker.start()

            if not self.timer:
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
        `pwargs` stands for 'process kwargs'
        """
        raise NotImplementedError

    def launch_thread(self, context: Context, *, twargs: K):
        """`twargs` as in keyword args made of a immutable tuple for a thread"""

        worker_fn = partial(self._operator_thread, twargs=twargs)

        self.thread_worker = ThreadWorkerInterface(self.bl_idname, worker_fn)
        self.thread_worker.start()

        if not self.timer:
            wm = self.wm(context)
            self.timer = wm.event_timer_add(
                time_step=_TIMER_INTERVAL_, window=context.window
            )
            wm.modal_handler_add(self)

    @staticmethod
    def _operator_thread(
        queue: queue.Queue[QT], cancel_event: threading.Event, *, twargs: K
    ):
        """
        `twargs` stands for both 'thread kwargs' and 'tuple kwargs'
        """
        raise NotImplementedError

    """instance properties backed by mangled class attributes"""

    @property
    def subprocess_worker(self) -> Optional[SubprocessWorkerInterface[QT]]:
        return self.__subprocess_worker

    @subprocess_worker.setter
    def subprocess_worker(self, wkr: Optional[SubprocessWorkerInterface[QT]]):
        if wkr is not None and self.__thread_worker is not None:
            raise DeveloperError("Cannot launch subprocess and thread simultaneously.")
        self.__subprocess_worker = wkr

    @property
    def thread_worker(self) -> Optional[ThreadWorkerInterface[QT]]:
        return self.__thread_worker

    @thread_worker.setter
    def thread_worker(self, wkr: Optional[ThreadWorkerInterface[QT]]):
        if wkr is not None and self.__subprocess_worker is not None:
            raise DeveloperError("Cannot launch subprocess and thread simultaneously.")
        self.__thread_worker = wkr

    @property
    def timer(self) -> Optional[Timer]:
        return self.__timer

    @timer.setter
    def timer(self, tmr: Optional[Timer]):
        self.__timer = tmr

    @property
    def data(self) -> MediaIOMetadata:
        """metadata property group as a dataclass."""
        if self.__data is None:
            raise DeveloperError("Metadata as dataclass not available in this scope.")
        else:
            return self.__data

    @data.setter
    def data(self, new_data: MediaIOMetadata) -> None:
        self.__data = new_data

    def wm(self, context: Context) -> WindowManager:
        if not (wm := context.window_manager):
            raise UnexpectedError("Poll-guard failed for window manager.")
        return wm

    def __load_global_data(self, prefs: Mosplat_AP_Global, props: Mosplat_PG_Global):
        # try to restore both dataclass and property group from local JSON
        # mangle so that only base class can access
        try:
            self.__data, load_msg = load_metadata_property_group_from_json(props, prefs)
            self.logger.info(load_msg)  # log the load message
        except UserWarning as e:  # raised if filesystem permission errors
            self.__data = None
            raise UserFacingError(
                "Operator cannot continue with insufficient permissions.", e
            ) from e

    @contextlib.contextmanager
    def SUBPROCESS_MANAGER(
        self, pwargs: K
    ) -> Generator[Tuple[Callable, Type[SubprocessWorkerInterface], K], Any, Any]:
        """
        to start subprocesses with `multiprocessing`, the passed function, arguments, and source module need to be pickled.
        this causes a problem when combined with the relative path import constraint that Blender enforces. the reason being is that Blender addons are dynamically imported in a namespace that the subprocess will not be able to find on disk, so subsequently resolved relative paths will include that namespace and error out.
        thus, we need to re-import the modules of all objects that need to be pickled using real, discoverable modules.
        furthermore, if we pop the imported modules within this context manager, we can effectively use `multiprocessing` while still maintaining best practices within Blender's development constraints.
        """
        meta = AddonMeta()

        # save state of sys modules beforehand
        old_modules_keys = set(sys.modules.keys())

        subproc_operator_import = meta.main_proc_import_to_subproc_import(
            self.__module__
        )
        operator_module: ModuleType = importlib.import_module(subproc_operator_import)

        subproc_worker_import = meta.main_proc_import_to_subproc_import(
            SubprocessWorkerInterface.__module__
        )
        worker_module: ModuleType = importlib.import_module(subproc_worker_import)
        worker_class_name = SubprocessWorkerInterface.__qualname__.rpartition(".")[-1]

        PwargsClass: Type[K] = type(pwargs)
        subproc_pwargs_import = meta.main_proc_import_to_subproc_import(
            PwargsClass.__module__
        )
        pwargs_module: ModuleType = importlib.import_module(subproc_pwargs_import)
        pwargs_class_name = PwargsClass.__qualname__.rpartition(".")[-1]

        imported_modules: List[ModuleType] = [
            operator_module,
            worker_module,
            pwargs_module,
        ]

        try:
            process_entrypoint: Callable = getattr(
                operator_module, OPERATOR_PROCESS_ENTRYPOINT_FN
            )
            assert isinstance(process_entrypoint, Callable)

            ImportedWorkerClass: Type[SubprocessWorkerInterface] = getattr(
                worker_module, worker_class_name
            )

            ImportedPwargsClass: Type[K] = getattr(pwargs_module, pwargs_class_name)

        except (AttributeError, AssertionError) as e:
            raise DeveloperError(
                f"Cannot create subprocess imports for '{self.bl_idname}'.", e
            ) from e

        I = TypeVar("I")

        def recreate_pwarg_item(item: I) -> I:
            item_class = type(item)
            mod_name = getattr(item_class, "__module__", "")
            subproc_import = meta.main_proc_import_to_subproc_import(mod_name)
            if mod_name == subproc_import:
                return item

            item_class_name: str = item_class.__qualname__.rpartition(".")[-1]

            mod: ModuleType = importlib.import_module(subproc_import)
            imported_modules.append(mod)

            try:
                ImportedItemClass = getattr(mod, item_class_name)
            except AttributeError as e:
                raise DeveloperError(
                    f"Cannot create subprocess imports for '{self.bl_idname}'.", e
                ) from e

            if isinstance(item, SupportsToFromDict):
                cls: SupportsToFromDict = ImportedItemClass
                dictionary = item.to_dict()
                new_item = cls.from_dict(dictionary)
                return cast(I, new_item)  # at runtime,
            else:
                # make the developer to-do known!
                raise DeveloperError(
                    f"Encountered a subprocess argument without safe import support: {item=}"
                )

        # rebuild `pwargs` items
        new_pwargs = ImportedPwargsClass._make(
            [recreate_pwarg_item(item) for item in pwargs]
        )

        yield (process_entrypoint, ImportedWorkerClass, new_pwargs)

        new_modules_keys = set(sys.modules.keys())

        # pop all imported modules
        addon_base_id = meta.base_id
        for key in new_modules_keys - old_modules_keys:
            if addon_base_id in key:  # do not pop standard lib modules
                sys.modules.pop(key)
