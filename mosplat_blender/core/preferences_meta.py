# auto-generated in build: 2026-02-01 03:57:23.861771 


from __future__ import annotations

Mosplat_AP_GlobalMeta = {
    'cache_dir': {'name': 'Cache Directory', 'description': 'Cache directory on disk used by the addon'},
    'json_log_subdir': {'name': 'JSON Log Cache Subdirectory', 'description': 'Subdirectory (relative to cache) for JSON logs'},
    'vggt_model_subdir': {'name': 'Model Cache Subdirectory', 'description': 'Subdirectory where the VGGT model data will be stored'},
    'data_output_path': {'name': 'Data Output Path', 'description': 'Output directory for processed data generated from the selected media directory.\nRelative paths are resolved against the selected media directory.\nThe token {media_directory_name} will be replaced with the base name of the selected media directory.'},
    'preprocess_media_script_file': {'name': 'Preprocess Media Script File', 'description': ''},
    'media_extensions': {'name': 'Media Extensions', 'description': 'Comma separated string of all file extensions that should be considered as media files within the media directory'},
    'max_frame_range': {'name': 'Max Frame Range', 'description': "The max frame range that can be processed at once.\nSet to '-1' for no limit.\nWARNING: Change this preference with knowledge of the capabilities of your own machine."},
    'json_log_filename_format': {'name': 'JSON Log Filename Format', 'description': 'strftime-compatible filename pattern'},
    'json_date_log_format': {'name': 'JSON Date Format', 'description': 'strftime format for JSON log timestamps'},
    'json_log_format': {'name': 'JSON Log Format', 'description': ''},
    'stdout_date_log_format': {'name': 'STDOUT Date Format', 'description': 'strftime format for console log timestamps'},
    'stdout_log_format': {'name': 'STDOUT Log Format', 'description': ''},
    'vggt_hf_id': {'name': 'VGGT Hugging Face ID', 'description': 'ID of VGGT pre-trained model on Hugging Face'},
}
