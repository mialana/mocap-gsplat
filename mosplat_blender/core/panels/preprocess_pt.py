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

        props = self.props(context)

        column.operator(OperatorIDEnum.INITIALIZE_MODEL)

        column.separator()

        box = column.box()
        box.label(text=OperatorIDEnum.label_factory(OperatorIDEnum.LOAD_IMAGES))
        op_row = box.row()
        op_row.operator(
            OperatorIDEnum.LOAD_IMAGES, text=props.current_image_dir, icon="FILE_FOLDER"
        )
