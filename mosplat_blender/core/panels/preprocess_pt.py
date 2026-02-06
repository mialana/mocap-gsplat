import os
from pathlib import Path

from core.panels.base_pt import MosplatPanelBase, column_factory
from core.properties import Mosplat_PG_MediaIODataset
from infrastructure.schemas import EnvVariableEnum, MediaFileStatus, OperatorIDEnum

_median_as_status: MediaFileStatus = MediaFileStatus(filepath="DIRECTORY MEDIANS")


class Mosplat_PT_Preprocess(MosplatPanelBase):
    def draw_with_layout(self, pkg, layout):
        props = pkg.props
        column = layout.column()
        init_model_box = column.box()
        init_model_box.row().operator(OperatorIDEnum.INITIALIZE_MODEL)
        progress = props.progress_accessor
        prog_curr: int = progress.current
        prog_total: int = progress.total
        if prog_curr > 0 and prog_total > 0:
            init_model_box.row().progress(factor=(float(prog_curr) / float(prog_total)))

        box = column.box()
        box.row().label(text=props._meta.current_media_dir.name)
        box.row().prop(props, props._meta.current_media_dir.id, text="")

        dataset = props.dataset_accessor

        statuses = dataset.collection_property_to_dataclass_list(
            dataset.statuses_accessor
        )
        statuses_length: int = len(statuses)
        if statuses_length == 0:
            return  # early return

        statuses_box = box.box()
        statuses_box.label(text=dataset._meta.media_file_statuses.name)

        grid = statuses_box.grid_flow(columns=4, align=True, row_major=True)

        if EnvVariableEnum.TESTING in os.environ:
            self._overwrite_median_as_status(dataset)
            statuses.append(_median_as_status)

        for s in statuses:
            _media_filename = Path(s.filepath).name
            _icon = "CON_TRANSFORM_CACHE" if s is _median_as_status else "FILE_MOVIE"
            column_factory(grid, _media_filename, not s.is_valid, icon=_icon)

            fc_matches, w_matches, h_matches = s.matches_dataset(dataset)

            column_factory(grid, f"F: {s.frame_count}", not fc_matches, pos="RIGHT")
            column_factory(grid, f"W: {s.width}", not w_matches, pos="RIGHT")
            column_factory(grid, f"H: {s.height}", not h_matches, pos="RIGHT")

        ranges = dataset.ranges_accessor
        if len(ranges) > 0:
            ranges_box = box.box()
            ranges_box.label(text=dataset._meta.processed_frame_ranges.name)
            for r in ranges:
                scripts = r.scripts_accessor
                has_scripts = len(scripts) > 0
                range_row = ranges_box.row()

                factor = 0.25 if has_scripts else 1
                split = range_row.split(factor=factor, align=True)
                split.label(
                    text=f"{str(r.start_frame)}-{str(r.end_frame)}",
                    icon="CON_ACTION",
                )

                if has_scripts:
                    script_column = split.column(align=True)
                    script_column.alignment = "EXPAND"
                    for script in scripts:
                        script_row = script_column.row()
                        script_row.alignment = "RIGHT"
                        script_row.label(text=Path(script.script_path).name)

        box.row().prop(props, props._meta.current_frame_range.id)
        box.row().operator(OperatorIDEnum.EXTRACT_FRAME_RANGE)

        column.row().operator(OperatorIDEnum.RUN_PREPROCESS_SCRIPT)

    @staticmethod
    def _overwrite_median_as_status(dataset: Mosplat_PG_MediaIODataset):
        global _median_as_status
        _median_as_status.overwrite(
            is_valid=dataset.is_valid_media_directory,
            frame_count=dataset.median_frame_count,
            width=dataset.median_width,
            height=dataset.median_height,
            mod_time=-1.0,
            file_size=-1,
        )
