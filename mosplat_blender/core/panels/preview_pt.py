from ...infrastructure.constants import SPLAT_PLAYER_OBJ_NAME
from .base_pt import MosplatPanelBase


class Mosplat_PT_preview(MosplatPanelBase):
    @classmethod
    def _contexted_poll(cls, pkg) -> bool:
        import bpy

        return bpy.data.objects.get(SPLAT_PLAYER_OBJ_NAME)

    def draw_with_layout(self, pkg, layout):
        props = pkg.props
        meta = props._meta

        layout.prop(props, meta.splat_render_mode.id)
