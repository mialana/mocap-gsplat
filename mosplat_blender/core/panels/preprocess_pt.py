import os
from pathlib import Path

from ...infrastructure.identifiers import OperatorIDEnum
from ...infrastructure.schemas import EnvVariableEnum, MediaFileStatus
from ..properties import Mosplat_PG_MediaIOMetadata
from .base_pt import MosplatPanelBase, column_factory

_median_as_status: MediaFileStatus = MediaFileStatus(filepath="DIRECTORY MEDIANS")


class Mosplat_PT_preprocess(MosplatPanelBase):
    def draw_with_layout(self, pkg, layout):
        props = pkg.props
        column = layout.column()

        box = column.box()
        box.row().label(text=props._meta.media_directory.name)
        box.row().prop(props, props._meta.media_directory.id, text="")

        data = props.metadata_accessor

        statuses = data.collection_property_to_dataclass_list(data.statuses_accessor)
        statuses_length: int = len(statuses)
        if statuses_length == 0:
            return  # early return

        statuses_box = box.box()
        statuses_box.label(text=data._meta.media_file_statuses.name)

        grid = statuses_box.grid_flow(columns=4, align=True, row_major=True)

        if EnvVariableEnum.TESTING in os.environ:
            self._overwrite_median_as_status(data)
            statuses.append(_median_as_status)

        for s in statuses:
            _media_filename = Path(s.filepath).name
            _icon = "CON_TRANSFORM_CACHE" if s is _median_as_status else "FILE_MOVIE"
            column_factory(grid, _media_filename, not s.is_valid, icon=_icon)

            fc_matches, w_matches, h_matches = s.matches_metadata(data)

            column_factory(grid, f"F: {s.frame_count}", not fc_matches, pos="RIGHT")
            column_factory(grid, f"W: {s.width}", not w_matches, pos="RIGHT")
            column_factory(grid, f"H: {s.height}", not h_matches, pos="RIGHT")

        ranges = data.ranges_accessor
        if len(ranges) > 0:
            ranges_box = box.box()
            ranges_box.label(text=data._meta.processed_frame_ranges.name)
            for r in ranges:
                script = r.script_accessor
                has_script: bool = script.file_path != ""
                range_row = ranges_box.row()

                factor = 0.25 if has_script else 1
                split = range_row.split(factor=factor, align=True)
                split.label(
                    text=f"{str(r.start_frame)}-{str(r.end_frame)}",
                    icon="CON_ACTION",
                )

                if has_script:
                    script_column = split.column(align=True)
                    script_column.alignment = "EXPAND"
                    script_row = script_column.row()
                    script_row.alignment = "RIGHT"
                    script_row.label(text=Path(script.file_path).name)

        box.row().prop(props, props._meta.frame_range.id)
        box.row().operator(OperatorIDEnum.EXTRACT_FRAME_RANGE)

        column.row().operator(OperatorIDEnum.RUN_PREPROCESS_SCRIPT)

    @staticmethod
    def _overwrite_median_as_status(data: Mosplat_PG_MediaIOMetadata):
        global _median_as_status
        _median_as_status.overwrite(
            is_valid=data.is_valid_media_directory,
            frame_count=data.median_frame_count,
            width=data.median_width,
            height=data.median_height,
            mod_time=-1.0,
            file_size=-1,
        )
