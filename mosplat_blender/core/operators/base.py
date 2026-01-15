import bpy
from bpy.types import Context, Event, WindowManager

from typing import Set, TYPE_CHECKING, TypeAlias

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


class MosplatOperatorBase(MosplatBlTypeMixin, bpy.types.Operator):
    id_enum_type = OperatorIDEnum

    @classmethod
    def at_registration(cls):
        super().at_registration()

        cls.bl_label = cls.bl_idname.replace(".", " ")

    @classmethod
    def poll(cls, context) -> bool:
        return bool(
            context.window_manager
            and check_props_safe(context)
            and check_prefs_safe(context)
        )

    def props(self, context: Context) -> Mosplat_PG_Global:
        return check_propertygroup(
            context.scene
        )  # let real runtimeerror rise as we trust poll to guard this call

    def prefs(self, context: Context) -> Mosplat_AP_Global:
        return check_addonpreferences(
            context.preferences
        )  # let real runtimeerror rise as we trust poll to guard this call

    def wm(self, context: Context) -> WindowManager:
        if not (wm := context.window_manager):
            raise RuntimeError("Something went wrong with `poll`-guard.")
        return wm

    """
    explictly re-define interface of these method signatures using the designated `TypeAlias`
    """

    def modal(self, context: Context, event: Event) -> OperatorReturnItemsSet: ...

    def invoke(self, context: Context, event: Event) -> OperatorReturnItemsSet: ...

    def execute(self, context: Context) -> OperatorReturnItemsSet: ...
