from ...infrastructure.constants import OperatorIDEnum, PanelIDEnum

from .base import MosplatPanelBase


class Mosplat_PT_Preprocess(MosplatPanelBase):
    bl_idname = PanelIDEnum.PREPROCESS

    bl_parent_id = PanelIDEnum.MAIN

    @classmethod
    def poll(cls, context):
        return True

    def draw_safe(self, context, layout):
        layout.row().operator(OperatorIDEnum.INITIALIZE_MODEL)

        layout.row().operator(OperatorIDEnum.LOAD_IMAGES)
