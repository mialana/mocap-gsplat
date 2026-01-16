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
        prefs = self.prefs(context)

        column.operator(OperatorIDEnum.INITIALIZE_MODEL)

        column.separator()

        box = column.box()
        box.row().label(text=props.bl_rna.properties["current_media_dir"].name)
        box.row().prop(props, "current_media_dir", text="")

        box.row().label(text=prefs.bl_rna.properties["data_output_path"].name)
        box.row().prop(prefs, "data_output_path", text="")

        box.row().label(
            text=prefs.bl_rna.properties["preprocess_media_script_file"].name
        )
        box.row().prop(prefs, "preprocess_media_script_file", text="")
