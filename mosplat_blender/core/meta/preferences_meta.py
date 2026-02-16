# 2026-02-16 03:21:44.967640
# created using 'generate_property_meta_files.py'


from typing import NamedTuple

from ...infrastructure.schemas import PropertyMeta


class Mosplat_AP_Global_Meta(NamedTuple):
    cache_dir: PropertyMeta
    cache_subdir_logs: PropertyMeta
    cache_subdir_model: PropertyMeta
    media_output_dir_format: PropertyMeta
    preprocess_media_script_file: PropertyMeta
    media_extensions: PropertyMeta
    max_frame_range: PropertyMeta
    json_log_filename_format: PropertyMeta
    json_date_log_format: PropertyMeta
    json_log_format: PropertyMeta
    stdout_date_log_format: PropertyMeta
    stdout_log_format: PropertyMeta
    blender_max_log_entries: PropertyMeta
    vggt_hf_id: PropertyMeta
    create_preview_images: PropertyMeta
    ply_file_format: PropertyMeta


MOSPLAT_AP_GLOBAL_META = Mosplat_AP_Global_Meta(
    cache_dir=PropertyMeta(
        id="cache_dir",
        name="Cache Directory",
        description="Cache directory on disk used by the addon",
    ),
    cache_subdir_logs=PropertyMeta(
        id="cache_subdir_logs",
        name="Logs Cache Subdirectory",
        description="Subdirectory in cache directory for JSON logs",
    ),
    cache_subdir_model=PropertyMeta(
        id="cache_subdir_model",
        name="Model Cache Subdirectory",
        description="Subdirectory in cache directory where model data will be stored",
    ),
    media_output_dir_format=PropertyMeta(
        id="media_output_dir_format",
        name="Media Output Directory Format",
        description="",
    ),
    preprocess_media_script_file=PropertyMeta(
        id="preprocess_media_script_file",
        name="Preprocess Media Script File",
        description="",
    ),
    media_extensions=PropertyMeta(
        id="media_extensions",
        name="Media Extensions",
        description="Comma separated string of all file extensions that should be considered as media files within the media directory",
    ),
    max_frame_range=PropertyMeta(
        id="max_frame_range", name="Max Frame Range", description=""
    ),
    json_log_filename_format=PropertyMeta(
        id="json_log_filename_format",
        name="JSON Logging Filename Format",
        description="strftime-compatible filename pattern",
    ),
    json_date_log_format=PropertyMeta(
        id="json_date_log_format",
        name="JSON Logging Date Format",
        description="strftime format for JSON log timestamps",
    ),
    json_log_format=PropertyMeta(
        id="json_log_format", name="JSON Logging Format", description=""
    ),
    stdout_date_log_format=PropertyMeta(
        id="stdout_date_log_format",
        name="STDOUT Logging Date Format",
        description="strftime format for console log timestamps",
    ),
    stdout_log_format=PropertyMeta(
        id="stdout_log_format", name="STDOUT Logging Log Format", description=""
    ),
    blender_max_log_entries=PropertyMeta(
        id="blender_max_log_entries",
        name="BLENDER Logging Max Entries",
        description="The max amount of log entries that will be stored as Blender data and displayed in UI.",
    ),
    vggt_hf_id=PropertyMeta(
        id="vggt_hf_id",
        name="VGGT Hugging Face ID",
        description="ID of VGGT pre-trained model on Hugging Face",
    ),
    create_preview_images=PropertyMeta(
        id="create_preview_images",
        name="Create Preview Images",
        description="Create preview images for 1. after raw frame extraction and 2. after running preprocess script. The images will be in the same directory as the binary processed data. Note that writing to disk is difficult to optimize and will cause a non-arbitrary increase in operation time.",
    ),
    ply_file_format=PropertyMeta(
        id="ply_file_format",
        name="PLY File Format",
        description="Format of outputted point cloud files after running model inference",
    ),
)
