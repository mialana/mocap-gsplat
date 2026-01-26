from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, Any

from ...infrastructure.schemas import PanelIDEnum, OperatorIDEnum

from .base_pt import MosplatPanelBase

if TYPE_CHECKING:
    from ..properties import Mosplat_PG_MediaFileStatus
else:
    Mosplat_PG_MediaFileStatus: TypeAlias = Any


class Mosplat_PT_Preprocess(MosplatPanelBase):
    bl_idname = PanelIDEnum.PREPROCESS
    bl_description = "Holds operations for preprocessing Mosplat data"

    bl_parent_id = PanelIDEnum.MAIN

    def draw_with_layout(self, context, layout):
        column = layout.column()

        props = self.props

        box = column.box()
        box.row().label(text=props.get_prop_name("current_media_dir"))
        box.row().prop(props, "current_media_dir", text="")

        if len(props.dataset_accessor.statuses_accessor) > 0:
            media_box = box.box()
            media_box.label(
                text=props.dataset_accessor.get_prop_name("media_file_statuses")
            )
            grid_flow = media_box.grid_flow(columns=4, align=True, row_major=True)
            for item in props.dataset_accessor.statuses_accessor:
                media = item
                media_filename = Path(media.filepath).name
                name_column = grid_flow.column()
                name_column.label(text=media_filename, icon="FILE_MOVIE")
                name_column.alert = not media.is_valid
                grid_flow.column().label(text=f"F: {str(media.frame_count)}")
                grid_flow.column().label(text=f"W: {str(media.width)}")
                grid_flow.column().label(text=f"H: {str(media.height)}")

        box.row().prop(props, "current_frame_range")

        column.row().operator(OperatorIDEnum.EXTRACT_FRAME_RANGE)

        if len(props.dataset_accessor.ranges_accessor) > 0:
            statuses_box = box.box()

        # column.operator(OperatorIDEnum.INITIALIZE_MODEL)

        # column.separator()

        # column.row().operator(OperatorIDEnum.RUN_INFERENCE)
