import bpy
from bpy.types import Context, Event, WindowManager

from typing import Set, TYPE_CHECKING, TypeAlias, ClassVar
from enum import Enum
from functools import partial

from ..checks import (
    check_addonpreferences,
    check_propertygroup,
    check_prefs_safe,
    check_props_safe,
)
from ..properties import Mosplat_PG_Global
from ..preferences import Mosplat_AP_Global
from ...infrastructure.mixins import MosplatBlTypeMixin
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


class MosplatOperatorBase(MosplatBlTypeMixin, bpy.types.Operator):
    bl_category = OperatorIDEnum._category()

    id_enum_type = OperatorIDEnum
    poll_reqs: ClassVar[Set[OperatorPollReqs]] = {
        OperatorPollReqs.PREFS,
        OperatorPollReqs.PROPS,
        OperatorPollReqs.WINDOW_MANAGER,
    }

    @classmethod
    def at_registration(cls):
        super().at_registration()

        if cls.guard_type_of_bl_idname(cls.bl_idname, cls.id_enum_type):
            cls.bl_label = OperatorIDEnum.label_factory(cls.bl_idname)

    @classmethod
    def poll(cls, context) -> bool:
        return all(req.value(cls, context) for req in cls.poll_reqs)

    def prefs(self, context: Context) -> Mosplat_AP_Global:
        return check_addonpreferences(
            context.preferences
        )  # let real runtimeerror rise as we trust poll to guard this call

    def props(self, context: Context) -> Mosplat_PG_Global:
        return check_propertygroup(
            context.scene
        )  # let real runtimeerror rise as we trust poll to guard this call

    def wm(self, context: Context) -> WindowManager:
        if not (wm := context.window_manager):
            raise RuntimeError("Something went wrong with `poll`-guard.")
        return wm

    """
    explictly re-define interface of these method signatures using the designated `TypeAlias`
    """

    def invoke(self, context: Context, event: Event) -> OperatorReturnItemsSet: ...

    def execute(self, context: Context) -> OperatorReturnItemsSet: ...

    def modal(self, context: Context, event: Event) -> OperatorReturnItemsSet: ...
