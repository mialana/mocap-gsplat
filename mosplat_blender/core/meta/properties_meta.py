# 2026-02-02 13:12:50.767790
# created using 'generate_property_meta_files.py'


from typing import NamedTuple

from ...infrastructure.schemas import PropertyMeta


class Mosplat_PG_AppliedPreprocessScript_Meta(NamedTuple):
    script_path: PropertyMeta
    mod_time: PropertyMeta
    file_size: PropertyMeta


class Mosplat_PG_ProcessedFrameRange_Meta(NamedTuple):
    start_frame: PropertyMeta
    end_frame: PropertyMeta
    applied_preprocess_scripts: PropertyMeta


class Mosplat_PG_MediaFileStatus_Meta(NamedTuple):
    filepath: PropertyMeta
    frame_count: PropertyMeta
    width: PropertyMeta
    height: PropertyMeta
    is_valid: PropertyMeta
    mod_time: PropertyMeta
    file_size: PropertyMeta


class Mosplat_PG_MediaIODataset_Meta(NamedTuple):
    base_directory: PropertyMeta
    is_valid_media_directory: PropertyMeta
    median_frame_count: PropertyMeta
    median_width: PropertyMeta
    median_height: PropertyMeta
    media_file_statuses: PropertyMeta
    processed_frame_ranges: PropertyMeta


class Mosplat_PG_OperatorProgress_Meta(NamedTuple):
    current: PropertyMeta
    total: PropertyMeta
    in_use: PropertyMeta


class Mosplat_PG_LogEntry_Meta(NamedTuple):
    level: PropertyMeta
    message: PropertyMeta
    session_index: PropertyMeta
    full_message: PropertyMeta


class Mosplat_PG_LogEntryHub_Meta(NamedTuple):
    logs: PropertyMeta
    logs_active_index: PropertyMeta
    logs_level_filter: PropertyMeta


class Mosplat_PG_Global_Meta(NamedTuple):
    current_media_dir: PropertyMeta
    current_frame_range: PropertyMeta
    current_media_io_dataset: PropertyMeta
    current_operator_progress: PropertyMeta
    current_log_entry_hub: PropertyMeta


MOSPLAT_PG_APPLIEDPREPROCESSSCRIPT_META = Mosplat_PG_AppliedPreprocessScript_Meta(
    script_path=PropertyMeta(id="script_path", name="Script Path", description=""),
    mod_time=PropertyMeta(id="mod_time", name="Modification Time", description=""),
    file_size=PropertyMeta(id="file_size", name="File Size", description=""),
)

MOSPLAT_PG_PROCESSEDFRAMERANGE_META = Mosplat_PG_ProcessedFrameRange_Meta(
    start_frame=PropertyMeta(id="start_frame", name="Start Frame", description=""),
    end_frame=PropertyMeta(id="end_frame", name="End Frame", description=""),
    applied_preprocess_scripts=PropertyMeta(
        id="applied_preprocess_scripts",
        name="Applied Preprocess Scripts",
        description="",
    ),
)

MOSPLAT_PG_MEDIAFILESTATUS_META = Mosplat_PG_MediaFileStatus_Meta(
    filepath=PropertyMeta(id="filepath", name="Filepath", description=""),
    frame_count=PropertyMeta(id="frame_count", name="Frame Count", description=""),
    width=PropertyMeta(id="width", name="Width", description=""),
    height=PropertyMeta(id="height", name="Height", description=""),
    is_valid=PropertyMeta(id="is_valid", name="Is Valid", description=""),
    mod_time=PropertyMeta(id="mod_time", name="Modification Time", description=""),
    file_size=PropertyMeta(id="file_size", name="File Size", description=""),
)

MOSPLAT_PG_MEDIAIODATASET_META = Mosplat_PG_MediaIODataset_Meta(
    base_directory=PropertyMeta(
        id="base_directory",
        name="Base Directory",
        description="Filepath to directory containing media files being processed.",
    ),
    is_valid_media_directory=PropertyMeta(
        id="is_valid_media_directory", name="Is Valid Media Directory", description=""
    ),
    median_frame_count=PropertyMeta(
        id="median_frame_count",
        name="Median Frame Count",
        description="Median frame count for all media files within the selected media directory.",
    ),
    median_width=PropertyMeta(
        id="median_width",
        name="Median Width",
        description="Median width for all media files within the selected media directory.",
    ),
    median_height=PropertyMeta(
        id="median_height",
        name="Median Height",
        description="Median height for all media files within the selected media directory.",
    ),
    media_file_statuses=PropertyMeta(
        id="media_file_statuses", name="Media File Statuses", description=""
    ),
    processed_frame_ranges=PropertyMeta(
        id="processed_frame_ranges", name="Processed Frame Ranges", description=""
    ),
)

MOSPLAT_PG_OPERATORPROGRESS_META = Mosplat_PG_OperatorProgress_Meta(
    current=PropertyMeta(
        id="current",
        name="Progress Current",
        description="Singleton current progress of operators.",
    ),
    total=PropertyMeta(
        id="total",
        name="Progress Total",
        description="Singleton total progress of operators.",
    ),
    in_use=PropertyMeta(
        id="in_use",
        name="Progress In Use",
        description="Whether any operator is 'using' the progress-related properties.",
    ),
)

MOSPLAT_PG_LOGENTRY_META = Mosplat_PG_LogEntry_Meta(
    level=PropertyMeta(id="level", name="Log Entry Level", description=""),
    message=PropertyMeta(id="message", name="Log Entry Message", description=""),
    session_index=PropertyMeta(
        id="session_index",
        name="Log Session Index",
        description="The self-stored index represented as a monotonic increasing index since the session start.",
    ),
    full_message=PropertyMeta(
        id="full_message",
        name="Log Entry Full Message",
        description="The property that is displayed in the dynamic tooltip while hovering on the log item.",
    ),
)

MOSPLAT_PG_LOGENTRYHUB_META = Mosplat_PG_LogEntryHub_Meta(
    logs=PropertyMeta(id="logs", name="Log Entries Data", description=""),
    logs_active_index=PropertyMeta(
        id="logs_active_index", name="Log Entries Active Index", description=""
    ),
    logs_level_filter=PropertyMeta(
        id="logs_level_filter", name="Log Entries Level Filter", description=""
    ),
)

MOSPLAT_PG_GLOBAL_META = Mosplat_PG_Global_Meta(
    current_media_dir=PropertyMeta(
        id="current_media_dir",
        name="Media Directory",
        description="Filepath to directory containing media files to be processed.",
    ),
    current_frame_range=PropertyMeta(
        id="current_frame_range",
        name="Frame Range",
        description="Start and end frame of data to be processed.",
    ),
    current_media_io_dataset=PropertyMeta(
        id="current_media_io_dataset",
        name="Media IO Dataset",
        description="Dataset for all media I/O operations",
    ),
    current_operator_progress=PropertyMeta(
        id="current_operator_progress", name="Current Operator Progress", description=""
    ),
    current_log_entry_hub=PropertyMeta(
        id="current_log_entry_hub", name="Current Log Entry Hub", description=""
    ),
)
