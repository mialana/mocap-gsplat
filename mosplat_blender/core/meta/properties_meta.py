# 2026-02-25 01:29:35.804010
# created using 'generate_property_meta_files.py'


from typing import NamedTuple

from ...infrastructure.schemas import PropertyMeta


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
    confidence_percentile: PropertyMeta
    inference_mode: PropertyMeta


class Mosplat_PG_SplatTrainingConfig_Meta(NamedTuple):
    steps: PropertyMeta
    lr: PropertyMeta
    sh_degree: PropertyMeta
    fuse_by_voxel: PropertyMeta
    init_tactics: PropertyMeta
    scene_size: PropertyMeta
    alpha_lambda: PropertyMeta
    opacity_lambda: PropertyMeta
    refine_start_step: PropertyMeta
    refine_end_step: PropertyMeta
    refine_interval: PropertyMeta
    refine_grow_threshold: PropertyMeta
    reset_opacity_interval: PropertyMeta
    revised_opacities_heuristic: PropertyMeta
    save_ply_interval: PropertyMeta
    increment_ply_file: PropertyMeta


class Mosplat_PG_AppliedPreprocessScript_Meta(NamedTuple):
    file_path: PropertyMeta
    mod_time: PropertyMeta
    file_size: PropertyMeta


class Mosplat_PG_ProcessedFrameRange_Meta(NamedTuple):
    start_frame: PropertyMeta
    end_frame: PropertyMeta
    applied_preprocess_script: PropertyMeta
    applied_model_options: PropertyMeta


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


class Mosplat_PG_Global_Meta(NamedTuple):
    media_directory: PropertyMeta
    frame_range: PropertyMeta
    was_frame_range_extracted: PropertyMeta
    was_frame_range_preprocessed: PropertyMeta
    ran_inference_on_frame_range: PropertyMeta
    operator_progress: PropertyMeta
    log_entry_hub: PropertyMeta
    vggt_model_options: PropertyMeta
    splat_training_config: PropertyMeta
    media_io_metadata: PropertyMeta


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
    confidence_percentile=PropertyMeta(
        id="confidence_percentile",
        name="Confidence",
        description="Minimum percentile for model-inferred confidence",
    ),
    inference_mode=PropertyMeta(
        id="inference_mode", name="Inference Mode", description=""
    ),
)

MOSPLAT_PG_SPLATTRAININGCONFIG_META = Mosplat_PG_SplatTrainingConfig_Meta(
    steps=PropertyMeta(id="steps", name="Steps", description=""),
    lr=PropertyMeta(
        id="lr",
        name="Learning Rates",
        description="Learning rates of model parameters defined in the following order: `means`, `scales`, `quats`, `opacities`, `sh0`, `shN`",
    ),
    sh_degree=PropertyMeta(
        id="sh_degree", name="Spherical Harmonics Degree", description=""
    ),
    fuse_by_voxel=PropertyMeta(
        id="fuse_by_voxel",
        name="Fuse By Voxel",
        description="Whether to cull initial point cloud points to a representative per voxel.",
    ),
    init_tactics=PropertyMeta(
        id="init_tactics", name="Initialization Tactics", description=""
    ),
    scene_size=PropertyMeta(
        id="scene_size",
        name="Scene Size",
        description="Number of cameras capturing the scene (read-only)",
    ),
    alpha_lambda=PropertyMeta(
        id="alpha_lambda",
        name="Alpha Lambda",
        description="Weighting of alpha values in loss computation",
    ),
    opacity_lambda=PropertyMeta(
        id="opacity_lambda",
        name="Opacity Lambda",
        description="Weighting of opacity values in loss computation. Ineffective if using `revised_opacities_heuristic`.",
    ),
    refine_start_step=PropertyMeta(
        id="refine_start_step",
        name="Refine Start Step",
        description="Step to begin refining the model through densification and pruning. Set to -1 to disable all densification and pruning.",
    ),
    refine_end_step=PropertyMeta(
        id="refine_end_step",
        name="Refine End Step",
        description="Step to stop refining the model through densification and pruning.",
    ),
    refine_interval=PropertyMeta(
        id="refine_interval",
        name="Refine Interval",
        description="The amount of steps in between each refinement",
    ),
    refine_grow_threshold=PropertyMeta(
        id="refine_grow_threshold",
        name="Refine Grow Threshold",
        description="Splats with image plane gradient above this value will be split/duplicated.",
    ),
    reset_opacity_interval=PropertyMeta(
        id="reset_opacity_interval",
        name="Reset Opacity Interval",
        description="The amount of steps in between resetting the splat `opacities` parameter. Set to -1 to disable.",
    ),
    revised_opacities_heuristic=PropertyMeta(
        id="revised_opacities_heuristic",
        name="Revised Opacity Heuristic",
        description="Whether to use revised `opacities` heuristic from arXiv:2404.06109",
    ),
    save_ply_interval=PropertyMeta(
        id="save_ply_interval",
        name="Save to PLY Interval",
        description="The amount of steps in between saving an evaluated PLY file to disk.",
    ),
    increment_ply_file=PropertyMeta(
        id="increment_ply_file",
        name="Increment PLY file",
        description="Whether to save separate PLY files with file names incremented by step number.",
    ),
)

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
    applied_model_options=PropertyMeta(
        id="applied_model_options", name="Applied Model Options", description=""
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

MOSPLAT_PG_GLOBAL_META = Mosplat_PG_Global_Meta(
    media_directory=PropertyMeta(
        id="media_directory",
        name="Media Directory",
        description="Filepath to directory containing media files to be processed.",
    ),
    frame_range=PropertyMeta(
        id="frame_range",
        name="Frame Range",
        description="Start and end (exclusive) frame of data to be processed.",
    ),
    was_frame_range_extracted=PropertyMeta(
        id="was_frame_range_extracted",
        name="Was Frame Range Extracted",
        description="Tracks whether the currently selected frame range extracted already.",
    ),
    was_frame_range_preprocessed=PropertyMeta(
        id="was_frame_range_preprocessed",
        name="Was Frame Range Preprocessed",
        description="Tracks whether the currently selected frame range has been preprocessed.",
    ),
    ran_inference_on_frame_range=PropertyMeta(
        id="ran_inference_on_frame_range",
        name="Was Frame Range Inferred",
        description="Tracks whether the currently selected frame range has had data inference ran on it already.",
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
    splat_training_config=PropertyMeta(
        id="splat_training_config", name="Splat Training Config", description=""
    ),
    media_io_metadata=PropertyMeta(
        id="media_io_metadata",
        name="Media IO Metadata",
        description="Metadata for all media I/O operations",
    ),
)
