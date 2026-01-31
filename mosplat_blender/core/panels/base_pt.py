from __future__ import annotations

from bpy.types import Panel, UILayout, Context, UIList

from typing import Literal, TYPE_CHECKING, Optional

from ..checks import check_addonpreferences, check_propertygroup
from ...infrastructure.mixins import (
    CtxPackage,
    MosplatContextAccessorMixin,
    MosplatEnforceAttributesMixin,
)
from ...infrastructure.schemas import PanelIDEnum, UserFacingError, UnexpectedError
from ...infrastructure.constants import _MISSING_


if TYPE_CHECKING:
    from bpy.stub_internal.rna_enums import IconItems


class MosplatUIListBase(UIList, MosplatEnforceAttributesMixin):
    bl_idname: str = _MISSING_


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
        try:
            check_addonpreferences(context.preferences)
            check_propertygroup(context.scene)
            return True
        except (UserFacingError, UnexpectedError) as e:
            return False

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
            self.logger.warning("Layout not available in draw function.")
            return

        self.draw_with_layout(self.package(context), layout)

    def draw_with_layout(self, pkg: CtxPackage, layout: UILayout) -> None:
        """layout will always exist with this function"""
        ...
