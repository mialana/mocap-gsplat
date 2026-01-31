from ...infrastructure.schemas import PanelIDEnum, OperatorIDEnum

from .base_pt import MosplatPanelBase


class Mosplat_PT_Main(MosplatPanelBase):
    bl_idname = PanelIDEnum.MAIN
    bl_description = "Main panel holding all Mosplat panels"
    bl_options = {"HIDE_HEADER"}
    bl_order = 9000

    def draw_with_layout(self, pkg, layout):
        layout.operator(OperatorIDEnum.OPEN_ADDON_PREFERENCES, icon="SETTINGS")

        layout.operator(OperatorIDEnum.REPORT_GLOBAL_MESSAGES, icon="SPEAKER")
