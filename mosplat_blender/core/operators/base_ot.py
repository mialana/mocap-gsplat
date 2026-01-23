import bpy
from bpy.types import Context, WindowManager, Timer, Event

from typing import (
    Set,
    TYPE_CHECKING,
    TypeAlias,
    Optional,
    Generic,
    TypeVar,
)
import contextlib

from ..checks import check_prefs_safe, check_props_safe
from ...infrastructure.mixins import (
    MosplatBlTypeMixin,
    MosplatPGAccessorMixin,
    MosplatAPAccessorMixin,
)

from ...infrastructure.schemas import PollGuardError, OperatorIDEnum
from ...interfaces.worker_interface import MosplatWorkerInterface

if TYPE_CHECKING:
    from bpy.stub_internal.rna_enums import (
        OperatorReturnItems as _OperatorReturnItemsSafe,
    )
    from ..preferences import Mosplat_AP_Global
    from ..properties import Mosplat_PG_Global
else:
    _OperatorReturnItemsSafe: TypeAlias = str
    Mosplat_AP_Global: TypeAlias = Any
    Mosplat_PG_Global: TypeAlias = Any

OperatorReturnItemsSet: TypeAlias = Set[_OperatorReturnItemsSafe]
OptionalOperatorReturnItemsSet: TypeAlias = Optional[OperatorReturnItemsSet]

Q = TypeVar("Q")  # the type of the elements in worker queue, if used


class MosplatOperatorBase(
    Generic[Q],
    MosplatBlTypeMixin,
    MosplatPGAccessorMixin,
    MosplatAPAccessorMixin,
    bpy.types.Operator,
):
    bl_category = OperatorIDEnum._category()
    __id_enum_type__ = OperatorIDEnum

    @classmethod
    def at_registration(cls):
        super().at_registration()

        if cls.guard_type_of_bl_idname(cls.bl_idname, cls.__id_enum_type__):
            cls.bl_label = OperatorIDEnum.label_factory(cls.bl_idname)

    @classmethod
    def poll(cls, context) -> bool:
        if (
            (prefs := check_prefs_safe(context)) is None
            or (props := check_props_safe(context)) is None
            or context.window_manager is not None
        ):
            return False
        return (
            ((((overrideable_return))))
            if ((((overrideable_return))) := cls.contexted_poll(context, prefs, props)) is not ...
            else True
        )

    @classmethod
    def contexted_poll(
        cls, context: Context, prefs: Mosplat_AP_Global, props: Mosplat_PG_Global
    ) -> bool:
        """an overrideable entrypoint for `poll` with access to prefs and props"""
        ...

    @property
    def worker(self) -> Optional[MosplatWorkerInterface[Q]]:
        return getattr(self, "_worker", None)
    
    @worker.setter
    def worker(self, wkr: Optional[MosplatWorkerInterface[Q]]):
        self._worker = wkr

    @property
    def timer(self) -> Optional[Timer]:
        return getattr(self, "_timer", None)
    
    @timer.setter
    def timer(self, tmr: Optional[Timer]):
        self._timer = tmr

    @property
    def wm(self) -> WindowManager:
        if not (wm := self.context.window_manager):
            raise PollGuardError
        return wm

    @contextlib.contextmanager
    def context_block(self, context: Context):
        self.context = context
        self.props.context = context
        self.prefs.context = context
        try:
            yield
        finally:
            self.context = None
            self.props.context = None
            self.prefs.context = None

    def execute(self, context) -> OperatorReturnItemsSet:
        with self.context_block(context):
            return self.contexted_execute(context)

    def contexted_execute(self, context: Context) -> OperatorReturnItemsSet:
        """
        an overrideable entrypoint for `execute` that ensures context is available as a property
        so that subsequently `_props` and `_prefs` properties can be accessed
        """
        ...

    def invoke(self, context, event) -> OperatorReturnItemsSet:
        with self.context_block(context):
            return self.contexted_invoke(context, event)

    def contexted_invoke(
        self, context: Context, event: Event
    ) -> OperatorReturnItemsSet:
        """
        an overrideable entrypoint for `invoke` that ensures context is available as a property
        so that subsequently `_props` and `_prefs` properties can be accessed
        """
        ...

    def modal(self, context, event) -> OperatorReturnItemsSet:
        with self.context_block(context):
            if event.type in {"RIGHTMOUSE", "ESC"}:
                self._cleanup(context)
                return {"CANCELLED"}
            elif event.type != "TIMER":
                return {"PASS_THROUGH"}

            return (
                optional_return
                if (optional_return := self.contexted_modal(context, event))
                else {"RUNNING_MODAL", "PASS_THROUGH"}
            )

    def contexted_modal(
        self, context: Context, event: Event
    ) -> OptionalOperatorReturnItemsSet:
        """an overrideable entrypoint that abstracts away shared return paths in `modal` (see above)"""
        ...

    def cancel(self, context):
        with self.context_block(context):
            self._cleanup(context)

    def _cleanup(self, context: Context):
        if self.timer:
            self.wm.event_timer_remove(self.timer)
            self.logger().debug("Timer cleaned up")
        if self.worker:
            self.worker.cleanup()
            self.logger().debug("Worker cleaned up")
