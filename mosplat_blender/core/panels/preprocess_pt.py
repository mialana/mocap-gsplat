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

        dataset = props.dataset_accessor
        statuses = dataset.statuses_accessor
        if len(statuses) > 0:
            statuses_box = box.box()
            statuses_box.label(text=dataset.get_prop_name("media_file_statuses"))
            ranges_grid = statuses_box.grid_flow(columns=4, align=True, row_major=True)
            for status in statuses:
                media_filename = Path(status.filepath).name
                name_column = ranges_grid.column()
                name_column.label(text=media_filename, icon="FILE_MOVIE")
                name_column.alert = not status.is_valid

                frame_count_column = ranges_grid.column()
                frame_count_column.alignment = "RIGHT"
                frame_count_column.label(text=f"F: {str(status.frame_count)}")

                width_column = ranges_grid.column()
                width_column.alignment = "RIGHT"
                width_column.label(text=f"W: {str(status.width)}")

                height_column = ranges_grid.column()
                height_column.alignment = "RIGHT"
                height_column.label(text=f"H: {str(status.height)}")

        ranges = dataset.ranges_accessor
        if len(ranges) > 0:
            ranges_box = box.box()
            ranges_box.label(text=dataset.get_prop_name("processed_frame_ranges"))
            for range in ranges:
                scripts = range.scripts_accessor
                has_scripts = len(scripts) > 0
                range_row = ranges_box.row()

                factor = 0.25 if has_scripts else 1
                split = range_row.split(factor=factor, align=True)
                split.label(
                    text=f"{str(range.start_frame)}-{str(range.end_frame)}",
                    icon="CON_ACTION",
                )

                if has_scripts:
                    script_column = split.column(align=True)
                    script_column.alignment = "EXPAND"
                    for script in scripts:
                        script_row = script_column.row()
                        script_row.alignment = "RIGHT"
                        script_row.label(text=Path(script.script_path).name)

        box.row().prop(props, "current_frame_range")
        box.row().operator(OperatorIDEnum.EXTRACT_FRAME_RANGE)

        # column.operator(OperatorIDEnum.INITIALIZE_MODEL)

        # column.separator()

        # column.row().operator(OperatorIDEnum.RUN_INFERENCE)
