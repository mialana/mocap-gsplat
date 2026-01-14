import bpy
from bpy.types import Panel, UILayout, Context

from typing import Union

from ..checks import check_props_safe, check_prefs_safe
from ..properties import Mosplat_PG_Global
from ..preferences import Mosplat_AP_Global

from ...infrastructure.mixins import MosplatBlTypeMixin
from ...infrastructure.constants import PanelIDEnum, ADDON_PANEL_CATEGORY


class MosplatPanelBase(MosplatBlTypeMixin, Panel):
    id_enum_type = PanelIDEnum

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = ADDON_PANEL_CATEGORY

    @classmethod
    def at_registration(cls):
        super().at_registration()
        if cls.check_bl_idname_type(cls.bl_idname, cls.id_enum_type):
            cls.bl_label = PanelIDEnum.label_factory(cls.bl_idname)

    def props(self, context: Context) -> Union[Mosplat_PG_Global, None]:
        return check_props_safe(context)

    def prefs(self, context: Context) -> Union[Mosplat_AP_Global, None]:
        return check_prefs_safe(context)

    def draw(self, context: Context):
        if not (layout := self.layout):
            return

        return self.draw_safe(context, layout)

    def draw_safe(self, context: Context, layout: UILayout):
        """layout will always exist with this function"""
        ...
