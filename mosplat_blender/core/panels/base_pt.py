from __future__ import annotations

from bpy.types import Panel, UILayout, Context, UIList

from typing import Literal, TYPE_CHECKING, Optional
from dataclasses import dataclass

from ..checks import check_addonpreferences, check_propertygroup
from ...infrastructure.mixins import CtxPackage, MosplatContextAccessorMixin
from ...infrastructure.schemas import (
    PanelIDEnum,
    UIListIDEnum,
    UserFacingError,
    UnexpectedError,
)


if TYPE_CHECKING:
    from bpy.stub_internal.rna_enums import IconItems


def column_factory(
    layout: UILayout,
    text: str,
    alert: bool = False,
    icon: Optional[IconItems] = None,
    pos: Optional[Literal["EXPAND", "LEFT", "CENTER", "RIGHT"]] = None,
) -> UILayout:
    col = layout.column()
    if pos is not None:
        col.alignment = pos
    col.alert = alert
    col.label(text=text, icon=icon) if icon else col.label(text=text)
    return col


class MosplatUIListBase(UIList, MosplatContextAccessorMixin):
    __id_enum_type__ = UIListIDEnum


@dataclass(frozen=True)
class MosplatPanelMetadata:
    bl_idname: PanelIDEnum
    bl_description: str

    @property
    def bl_label(self) -> str:
        return PanelIDEnum.label_factory(self.bl_idname)

    bl_category: str = PanelIDEnum._category()
    bl_parent_id: Optional[PanelIDEnum] = None


class MosplatPanelBase(Panel, MosplatContextAccessorMixin[MosplatPanelMetadata]):
    __id_enum_type__ = PanelIDEnum

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = PanelIDEnum._category()

    @classmethod
    def poll(cls, context) -> bool:
        try:
            check_addonpreferences(context.preferences)
            check_propertygroup(context.scene)
            return True
        except (UserFacingError, UnexpectedError) as e:
            return False

    def draw(self, context: Context) -> None:
        if not (layout := self.layout):
            self.logger.warning("Layout not available in draw function.")
            return

        self.draw_with_layout(self.package(context), layout)

    def draw_with_layout(self, pkg: CtxPackage, layout: UILayout) -> None:
        """layout will always exist with this function"""
        ...
