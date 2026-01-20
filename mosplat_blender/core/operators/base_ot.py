import bpy
from bpy.types import Context, WindowManager

from typing import Set, TYPE_CHECKING, TypeAlias, ClassVar, Union
from enum import Enum
from functools import partial
from queue import Queue
from threading import Thread

from ..checks import (
    check_prefs_safe,
    check_props_safe,
)
from ...infrastructure.mixins import (
    MosplatBlTypeMixin,
    MosplatPGAccessorMixin,
    MosplatAPAccessorMixin,
)
from ...infrastructure.constants import OperatorIDEnum


if TYPE_CHECKING:
    from bpy.stub_internal.rna_enums import (
        OperatorReturnItems as _OperatorReturnItemsSafe,
    )
else:
    _OperatorReturnItemsSafe: TypeAlias = str

OperatorReturnItemsSet: TypeAlias = Set[_OperatorReturnItemsSafe]


class OperatorPollReqs(Enum):
    """Custom enum in case operator does not require use of one poll requirement"""

    PREFS = partial(lambda cls, context: check_prefs_safe(context))
    PROPS = partial(lambda cls, context: check_props_safe(context))
    WINDOW_MANAGER = partial(
        lambda cls, context: getattr(context, "window_manager", None)
    )


class MosplatOperatorBase(
    MosplatBlTypeMixin,
    MosplatPGAccessorMixin,
    MosplatAPAccessorMixin,
    bpy.types.Operator,
):
    bl_category = OperatorIDEnum._category()

    __id_enum_type__ = OperatorIDEnum
    __poll_reqs__: ClassVar[Union[Set[OperatorPollReqs], None]] = {
        OperatorPollReqs.PREFS,
        OperatorPollReqs.PROPS,
        OperatorPollReqs.WINDOW_MANAGER,
    }

    @classmethod
    def at_registration(cls):
        super().at_registration()

        if cls.guard_type_of_bl_idname(cls.bl_idname, cls.__id_enum_type__):
            cls.bl_label = OperatorIDEnum.label_factory(cls.bl_idname)

    @classmethod
    def poll(cls, context) -> bool:
        return (
            all(req.value(cls, context) for req in cls.__poll_reqs__)
            if cls.__poll_reqs__
            else True
        )

    @staticmethod
    def wm(context: Context) -> WindowManager:
        if not (wm := context.window_manager):
            raise RuntimeError("Something went wrong with `poll`-guard.")
        return wm

    def cancel(self, context):
        self._cleanup(context)

    def _cleanup(self, context: Context):
        if hasattr(self, "_timer"):
            if self._timer:
                self.wm(context).event_timer_remove(self._timer)
            self._timer = None
            self.logger().debug("Timer cleaned up")

        if hasattr(self, "_thread"):
            self._thread = None
            self.logger().debug("Thread cleaned up")

        if hasattr(self, "_queue"):
            queue = getattr(self, "_queue")
            if isinstance(queue, Queue):
                # drain queue
                while not queue.empty():
                    queue.get_nowait()

            self.logger().debug("Queue cleaned up")
