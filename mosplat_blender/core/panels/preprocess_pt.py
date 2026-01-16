from ...infrastructure.constants import OperatorIDEnum, PanelIDEnum

from .base_pt import MosplatPanelBase


class Mosplat_PT_Preprocess(MosplatPanelBase):
    bl_idname = PanelIDEnum.PREPROCESS
    bl_description = "Holds operations for preprocessing Mosplat data"

    bl_parent_id = PanelIDEnum.MAIN

    @classmethod
    def poll(cls, context):
        return True

    def draw_with_layout(self, context, layout):
        column = layout.column()
        # column.operator_context = "INVOKE_DEFAULT"  # all ops invoke before excecute

        column.operator(OperatorIDEnum.INITIALIZE_MODEL)
        column.operator(OperatorIDEnum.LOAD_IMAGES)
