from ...infrastructure.constants import OperatorIDEnum, PanelIDEnum

from .base_pt import MosplatPanelBase, PanelPollReqs


class Mosplat_PT_Preprocess(MosplatPanelBase):
    bl_idname = PanelIDEnum.PREPROCESS
    bl_description = "Holds operations for preprocessing Mosplat data"

    bl_parent_id = PanelIDEnum.MAIN

    poll_reqs = {PanelPollReqs.PROPS}

    def draw_with_layout(self, context, layout):
        column = layout.column()

        props = self.props(context)
        prefs = self.prefs(context)

        column.operator(OperatorIDEnum.INITIALIZE_MODEL)

        column.separator()

        box = column.box()
        box.row().label(text=props.bl_rna.properties["current_media_dir"].name)
        box.row().prop(props, "current_media_dir", text="")
