from bpy.types import Context, WindowManager, Timer, Event, Operator

from typing import Set, TYPE_CHECKING, TypeAlias, Optional, Generic, TypeVar, Any

from ..checks import check_prefs_safe, check_props_safe
from ...infrastructure.mixins import (
    MosplatBlTypeMixin,
    MosplatPGAccessorMixin,
    MosplatAPAccessorMixin,
    MosplatEncapsulatedContextMixin,
)

from ...infrastructure.schemas import (
    UnexpectedError,
    OperatorIDEnum,
    MediaIOMetadata,
    DeveloperError,
)
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
    Operator,
    Generic[Q],
    MosplatBlTypeMixin,
    MosplatPGAccessorMixin,
    MosplatAPAccessorMixin,
    MosplatEncapsulatedContextMixin,
):
    bl_category = OperatorIDEnum._category()
    __id_enum_type__ = OperatorIDEnum

    __worker: Optional[MosplatWorkerInterface[Q]] = None
    __timer: Optional[Timer] = None
    __metadata_dc: Optional[MediaIOMetadata] = None

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
            or context.window_manager is None
        ):
            return False
        return (
            overrideable_return
            if (overrideable_return := cls.contexted_poll(context, prefs, props))
            is not None
            else True  # if not implemented return true
        )

    @classmethod
    def contexted_poll(
        cls, context: Context, prefs: Mosplat_AP_Global, props: Mosplat_PG_Global
    ) -> bool:
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
    def metadata_dc(self) -> MediaIOMetadata:
        if self.__metadata_dc is None:
            raise DeveloperError("Metadata as dataclass not available in this scope.")
        else:
            return self.__metadata_dc

    @metadata_dc.setter
    def metadata_dc(self, mta: Optional[MediaIOMetadata]):
        self.__metadata_dc = mta

    @property
    def wm(self) -> WindowManager:
        if not (wm := self.context.window_manager):
            raise UnexpectedError("Poll-guard failed for window manager.")
        return wm

    def invoke(self, context, event) -> OperatorReturnItemsSet:
        with self.encapsulated_context_block(context):
            return (
                overrideable_return
                if (overrideable_return := self.contexted_invoke(context, event))
                is not None
                else self.execute(context)  # if not implemented return execute
            )

    def contexted_invoke(
        self, context: Context, event: Event
    ) -> OperatorReturnItemsSet:
        """
        an overrideable entrypoint for `execute` that ensures context is available as a property
        so that subsequently `props` and `prefs` properties can be accessed
        """
        ...

    def execute(self, context) -> OperatorReturnItemsSet:
        with self.encapsulated_context_block(context):
            self.metadata_dc = (
                self.props.metadata_ptr.to_dataclass()
            )  # set metadata before
            return self.contexted_execute(context)

    def contexted_execute(self, context: Context) -> OperatorReturnItemsSet:
        """
        an overrideable entrypoint for `execute` that ensures context is available as a property
        so that subsequently `props` and `prefs` properties can be accessed
        """
        ...

    def modal(self, context, event) -> OperatorReturnItemsSet:
        with self.encapsulated_context_block(context):
            if event.type in {"RIGHTMOUSE", "ESC"}:
                self.cleanup(context)
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
        while self.worker and (next := self.worker.dequeue()) is not None:
            try:
                return self.queue_callback(context, event, next)
            finally:
                if context.area:
                    context.area.tag_redraw()  # redraw UI

    def queue_callback(
        self, context: Context, event: Event, next: Q
    ) -> OptionalOperatorReturnItemsSet:
        """an entrypoint for when a new element is placed in the queue during `modal`"""
        ...

    def cancel(self, context):
        with self.encapsulated_context_block(context):
            self.cleanup(context)

    def cleanup(self, context: Context):
        # update JSON with current state of PG as source of truth
        metadata_json_filepath = self.props.metadata_json_filepath(self.prefs)
        self.props.metadata_ptr.to_JSON(metadata_json_filepath)

        if self.timer:
            self.wm.event_timer_remove(self.timer)
            self.timer = None
            self.logger().debug("Timer cleaned up")
        if self.worker:
            self.worker.cleanup()
            self.worker = None
            self.logger().debug("Worker cleaned up")

        self.metadata_dc = None  # metadata is not guaranteed to be in-sync anymore
