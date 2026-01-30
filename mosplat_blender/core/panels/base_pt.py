from __future__ import annotations

from bpy.types import Panel, UILayout, Context

from typing import Literal, TYPE_CHECKING, Optional

from ..checks import check_prefs_safe, check_props_safe
from ...infrastructure.mixins import CtxPackage, MosplatContextAccessorMixin
from ...infrastructure.schemas import PanelIDEnum


if TYPE_CHECKING:
    from bpy.stub_internal.rna_enums import IconItems


class MosplatPanelBase(Panel, MosplatContextAccessorMixin):
    __id_enum_type__ = PanelIDEnum

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = PanelIDEnum._category()

    @classmethod
    def at_registration(cls):
        super().at_registration()
        if cls.guard_type_of_bl_idname(cls.bl_idname, cls.__id_enum_type__):
            cls.bl_label = PanelIDEnum.label_factory(cls.bl_idname)

    @classmethod
    def poll(cls, context) -> bool:
        return (
            check_prefs_safe(context) is not None
            and check_props_safe(context) is not None
        )

    @staticmethod
    def _col_factory(
        layout: UILayout,
        text: str,
        alert: bool = False,
        icon: Optional[IconItems] = None,
        pos: Optional[Literal["EXPAND", "LEFT", "CENTER", "RIGHT"]] = None,
    ):
        col = layout.column()
        if pos is not None:
            col.alignment = pos
        col.alert = alert
        col.label(text=text, icon=icon) if icon else col.label(text=text)

    def draw(self, context: Context) -> None:
        if not (layout := self.layout):
            return

        self.draw_with_layout(self.package(context), layout)

    def draw_with_layout(self, pkg: CtxPackage, layout: UILayout) -> None:
        """layout will always exist with this function"""
        ...
