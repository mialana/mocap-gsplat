from ...infrastructure.constants import PanelIDEnum, OperatorIDEnum

from .base_pt import MosplatPanelBase


class Mosplat_PT_Main(MosplatPanelBase):
    bl_idname = PanelIDEnum.MAIN
    bl_description = "Main panel holding all Mosplat panels"

    poll_reqs = None

    def draw_with_layout(self, context, layout):
        layout.row().operator(OperatorIDEnum.OPEN_ADDON_PREFERENCES)
