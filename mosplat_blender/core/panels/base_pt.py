from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, List, Literal, Optional, Set, TypeAlias

from bpy.types import Context, Panel, UILayout, UIList

from ...infrastructure.identifiers import PanelIDEnum
from ...infrastructure.mixins import ContextAccessorMixin, CtxPackage
from ...infrastructure.schemas import UnexpectedError, UserFacingError
from ..checks import check_addonpreferences, check_propertygroup

if TYPE_CHECKING:
    from bpy.stub_internal.rna_enums import IconItems

    PanelTypeFlagItems: TypeAlias = Literal[  # stubs don't supply as an enum yet
        "DEFAULT_CLOSED", "HIDE_HEADER", "INSTANCED", "HEADER_LAYOUT_EXPAND"
    ]


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


@dataclass
class MosplatUIListMetadata:
    bl_idname: str
    bl_description: str


class MosplatUIListBase(UIList, ContextAccessorMixin[MosplatUIListMetadata]):
    pass


@dataclass
class MosplatPanelMetadata:
    bl_idname: str
    bl_description: str
    bl_label: str
    bl_parent_id: str
    bl_options: Set[PanelTypeFlagItems]
    bl_category: str = PanelIDEnum._category()
    bl_space_type: str = "VIEW_3D"
    bl_region_type: str = "UI"

    def __init__(
        self,
        *,
        bl_idname: PanelIDEnum,
        bl_description: str,
        bl_parent_id: Optional[PanelIDEnum] = None,
        bl_options: Set[PanelTypeFlagItems] = set(),
    ):
        self.bl_idname = bl_idname.value
        self.bl_description = bl_description
        self.bl_label = PanelIDEnum.label_factory(bl_idname)
        self.bl_parent_id = bl_parent_id.value if bl_parent_id else ""
        self.bl_options = bl_options


class MosplatPanelBase(Panel, ContextAccessorMixin[MosplatPanelMetadata]):
    @classmethod
    def poll(cls, context) -> bool:
        try:
            return cls._contexted_poll(cls.package(context))
        except (UserFacingError, UnexpectedError) as e:
            cls.class_logger.error(str(e))
            return False

    @classmethod
    def _contexted_poll(cls, pkg: CtxPackage) -> bool:
        """an overrideable entrypoint for `poll` with access to prefs and props"""
        return True  # if not overriden will return true

    def draw(self, context: Context) -> None:
        if not (layout := self.layout):
            self.logger.warning("Layout not available in draw function.")
            return

        self.draw_with_layout(self.package(context), layout)

    def draw_with_layout(self, pkg: CtxPackage, layout: UILayout) -> None:
        """layout will always exist with this function"""
        ...
