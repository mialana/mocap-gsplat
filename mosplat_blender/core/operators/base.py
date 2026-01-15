import bpy
from bpy.types import Context, Event, WindowManager

from typing import Union, Set, TYPE_CHECKING, TypeAlias

from ..checks import check_props_safe, check_prefs_safe
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

    def props(self, context: Context) -> Union[Mosplat_PG_Global, None]:
        return check_props_safe(context)

    def prefs(self, context: Context) -> Union[Mosplat_AP_Global, None]:
        return check_prefs_safe(context)

    """
    here we change the signature of overriden methods.
    we also create convenience methods for having the window manager available.
    they are opt-in, as if we do not need the window manager,
    just override the normal method.
    """

    def modal(self, context: Context, event: Event) -> OperatorReturnItemsSet:
        if not (wm := context.window_manager):
            return {"CANCELLED"}

        return self.execute_with_window_manager(context, wm)

    def invoke(self, context: Context, event: Event) -> OperatorReturnItemsSet:
        if not (wm := context.window_manager):
            return {"CANCELLED"}

        return self.invoke_with_window_manager(context, event, wm)

    def execute(self, context: Context) -> OperatorReturnItemsSet:
        if not (wm := context.window_manager):
            return {"CANCELLED"}

        return self.execute_with_window_manager(context, wm)

    def modal_with_window_manager(
        self, context: Context, event: Event, wm: WindowManager
    ) -> OperatorReturnItemsSet: ...

    def invoke_with_window_manager(
        self, context: Context, event: Event, wm: WindowManager
    ) -> OperatorReturnItemsSet: ...

    def execute_with_window_manager(
        self, context: Context, wm: WindowManager
    ) -> OperatorReturnItemsSet: ...
