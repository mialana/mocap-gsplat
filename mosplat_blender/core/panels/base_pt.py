from bpy.types import Panel, UILayout, Context

from ..checks import check_prefs_safe, check_props_safe
from ...infrastructure.mixins import (
    MosplatBlTypeMixin,
    MosplatPGAccessorMixin,
    MosplatAPAccessorMixin,
    MosplatEncapsulatedContextMixin,
)
from ...infrastructure.schemas import PanelIDEnum


class MosplatPanelBase(
    Panel,
    MosplatEncapsulatedContextMixin,
    MosplatBlTypeMixin,
    MosplatPGAccessorMixin,
    MosplatAPAccessorMixin,
):
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

    def draw(self, context: Context) -> None:
        if not (layout := self.layout):
            return

        with self.encapsulated_context_block(context):
            self.draw_with_layout(context, layout)

    def draw_with_layout(self, context: Context, layout: UILayout) -> None:
        """layout will always exist with this function"""
        ...
