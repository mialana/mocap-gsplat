# 2026-02-01 22:55:13.287558
# created using 'generate_property_meta_files.py'


from typing import NamedTuple

from ...infrastructure.schemas import PropertyMeta


class Mosplat_AP_Global_Meta(NamedTuple):
    cache_dir: PropertyMeta
    json_log_subdir: PropertyMeta
    vggt_model_subdir: PropertyMeta
    data_output_path: PropertyMeta
    preprocess_media_script_file: PropertyMeta
    media_extensions: PropertyMeta
    max_frame_range: PropertyMeta
    json_log_filename_format: PropertyMeta
    json_date_log_format: PropertyMeta
    json_log_format: PropertyMeta
    stdout_date_log_format: PropertyMeta
    stdout_log_format: PropertyMeta
    vggt_hf_id: PropertyMeta


MOSPLAT_AP_GLOBAL_META = Mosplat_AP_Global_Meta(
    cache_dir=PropertyMeta(
        id="cache_dir",
        name="Cache Directory",
        description="Cache directory on disk used by the addon",
    ),
    json_log_subdir=PropertyMeta(
        id="json_log_subdir",
        name="JSON Log Cache Subdirectory",
        description="Subdirectory (relative to cache) for JSON logs",
    ),
    vggt_model_subdir=PropertyMeta(
        id="vggt_model_subdir",
        name="Model Cache Subdirectory",
        description="Subdirectory where the VGGT model data will be stored",
    ),
    data_output_path=PropertyMeta(
        id="data_output_path", name="Data Output Path", description=""
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
        name="JSON Log Filename Format",
        description="strftime-compatible filename pattern",
    ),
    json_date_log_format=PropertyMeta(
        id="json_date_log_format",
        name="JSON Date Format",
        description="strftime format for JSON log timestamps",
    ),
    json_log_format=PropertyMeta(
        id="json_log_format", name="JSON Log Format", description=""
    ),
    stdout_date_log_format=PropertyMeta(
        id="stdout_date_log_format",
        name="STDOUT Date Format",
        description="strftime format for console log timestamps",
    ),
    stdout_log_format=PropertyMeta(
        id="stdout_log_format", name="STDOUT Log Format", description=""
    ),
    vggt_hf_id=PropertyMeta(
        id="vggt_hf_id",
        name="VGGT Hugging Face ID",
        description="ID of VGGT pre-trained model on Hugging Face",
    ),
)
