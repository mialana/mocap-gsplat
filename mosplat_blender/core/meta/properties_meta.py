# 2026-02-12 09:15:55.452178
# created using 'generate_property_meta_files.py'


from typing import NamedTuple

from mosplat_blender.infrastructure.schemas import PropertyMeta


class Mosplat_PG_AppliedPreprocessScript_Meta(NamedTuple):
    file_path: PropertyMeta
    mod_time: PropertyMeta
    file_size: PropertyMeta


class Mosplat_PG_ProcessedFrameRange_Meta(NamedTuple):
    start_frame: PropertyMeta
    end_frame: PropertyMeta
    applied_preprocess_script: PropertyMeta


class Mosplat_PG_MediaFileStatus_Meta(NamedTuple):
    filepath: PropertyMeta
    frame_count: PropertyMeta
    width: PropertyMeta
    height: PropertyMeta
    is_valid: PropertyMeta
    mod_time: PropertyMeta
    file_size: PropertyMeta


class Mosplat_PG_MediaIOMetadata_Meta(NamedTuple):
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


class Mosplat_PG_VGGTModelOptions_Meta(NamedTuple):
    inference_mode: PropertyMeta
    confidence_percentile: PropertyMeta
    enable_black_mask: PropertyMeta
    enable_white_mask: PropertyMeta


class Mosplat_PG_Global_Meta(NamedTuple):
    media_directory: PropertyMeta
    frame_range: PropertyMeta
    was_frame_range_extracted: PropertyMeta
    operator_progress: PropertyMeta
    log_entry_hub: PropertyMeta
    vggt_model_options: PropertyMeta
    media_io_metadata: PropertyMeta


MOSPLAT_PG_APPLIEDPREPROCESSSCRIPT_META = Mosplat_PG_AppliedPreprocessScript_Meta(
    file_path=PropertyMeta(id="file_path", name="File Path", description=""),
    mod_time=PropertyMeta(id="mod_time", name="Modification Time", description=""),
    file_size=PropertyMeta(id="file_size", name="File Size", description=""),
)

MOSPLAT_PG_PROCESSEDFRAMERANGE_META = Mosplat_PG_ProcessedFrameRange_Meta(
    start_frame=PropertyMeta(id="start_frame", name="Start Frame", description=""),
    end_frame=PropertyMeta(id="end_frame", name="End Frame", description=""),
    applied_preprocess_script=PropertyMeta(
        id="applied_preprocess_script", name="Applied Preprocess Script", description=""
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

MOSPLAT_PG_MEDIAIOMETADATA_META = Mosplat_PG_MediaIOMetadata_Meta(
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

MOSPLAT_PG_VGGTMODELOPTIONS_META = Mosplat_PG_VGGTModelOptions_Meta(
    inference_mode=PropertyMeta(
        id="inference_mode", name="Inference Mode", description=""
    ),
    confidence_percentile=PropertyMeta(
        id="confidence_percentile", name="Confidence Percentile", description=""
    ),
    enable_black_mask=PropertyMeta(
        id="enable_black_mask", name="Enable Black Mask", description=""
    ),
    enable_white_mask=PropertyMeta(
        id="enable_white_mask", name="Enable White Mask", description=""
    ),
)

MOSPLAT_PG_GLOBAL_META = Mosplat_PG_Global_Meta(
    media_directory=PropertyMeta(
        id="media_directory",
        name="Media Directory",
        description="Filepath to directory containing media files to be processed.",
    ),
    frame_range=PropertyMeta(
        id="frame_range",
        name="Frame Range",
        description="Start and end frame of data to be processed.",
    ),
    was_frame_range_extracted=PropertyMeta(
        id="was_frame_range_extracted",
        name="Was Frame Range Extracted",
        description="Tracks whether the currently selected frame range extracted already.",
    ),
    operator_progress=PropertyMeta(
        id="operator_progress", name="Operator Progress", description=""
    ),
    log_entry_hub=PropertyMeta(
        id="log_entry_hub", name="Log Entry Hub", description=""
    ),
    vggt_model_options=PropertyMeta(
        id="vggt_model_options", name="VGGT Model Options", description=""
    ),
    media_io_metadata=PropertyMeta(
        id="media_io_metadata",
        name="Media IO Metadata",
        description="Metadata for all media I/O operations",
    ),
)
