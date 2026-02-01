# 2026-02-01 18:45:41.541658
# created using 'generate_property_meta_files.py'

Mosplat_PG_AppliedPreprocessScript_Meta: dict[str, dict[str, str]] = {
    "script_path": {"name": "Script Path", "description": ""},
    "mod_time": {"name": "Modification Time", "description": ""},
    "file_size": {"name": "File Size", "description": ""},
}

Mosplat_PG_ProcessedFrameRange_Meta: dict[str, dict[str, str]] = {
    "start_frame": {"name": "Start Frame", "description": ""},
    "end_frame": {"name": "End Frame", "description": ""},
    "applied_preprocess_scripts": {
        "name": "Applied Preprocess Scripts",
        "description": "",
    },
}

Mosplat_PG_MediaFileStatus_Meta: dict[str, dict[str, str]] = {
    "filepath": {"name": "Filepath", "description": ""},
    "frame_count": {"name": "Frame Count", "description": ""},
    "width": {"name": "Width", "description": ""},
    "height": {"name": "Height", "description": ""},
    "is_valid": {"name": "Is Valid", "description": ""},
    "mod_time": {"name": "Modification Time", "description": ""},
    "file_size": {"name": "File Size", "description": ""},
}

Mosplat_PG_MediaIODataset_Meta: dict[str, dict[str, str]] = {
    "base_directory": {
        "name": "Base Directory",
        "description": "Filepath to directory containing media files being processed.",
    },
    "is_valid_media_directory": {"name": "Is Valid Media Directory", "description": ""},
    "median_frame_count": {
        "name": "Median Frame Count",
        "description": "Median frame count for all media files within the selected media directory.",
    },
    "median_width": {
        "name": "Median Width",
        "description": "Median width for all media files within the selected media directory.",
    },
    "median_height": {
        "name": "Median Height",
        "description": "Median height for all media files within the selected media directory.",
    },
    "media_file_statuses": {"name": "Media File Statuses", "description": ""},
    "processed_frame_ranges": {"name": "Processed Frame Ranges", "description": ""},
}

Mosplat_PG_OperatorProgress_Meta: dict[str, dict[str, str]] = {
    "current": {
        "name": "Progress Current",
        "description": "Singleton current progress of operators.",
    },
    "total": {
        "name": "Progress Total",
        "description": "Singleton total progress of operators.",
    },
    "in_use": {
        "name": "Progress In Use",
        "description": "Whether any operator is 'using' the progress-related properties.",
    },
}

Mosplat_PG_LogEntry_Meta: dict[str, dict[str, str]] = {
    "level": {"name": "Log Entry Level", "description": ""},
    "message": {"name": "Log Entry Message", "description": ""},
    "full_message": {
        "name": "Log Entry Full Message",
        "description": "The property that is displayed in the dynamic tooltip while hovering on the item.",
    },
}

Mosplat_PG_LogEntryHub_Meta: dict[str, dict[str, str]] = {
    "logs": {"name": "Log Entries Data", "description": ""},
    "logs_active_index": {"name": "Log Entries Active Index", "description": ""},
    "logs_level_filter": {"name": "Log Entries Level Filter", "description": ""},
}

Mosplat_PG_Global_Meta: dict[str, dict[str, str]] = {
    "current_media_dir": {
        "name": "Media Directory",
        "description": "Filepath to directory containing media files to be processed.",
    },
    "current_frame_range": {
        "name": "Frame Range",
        "description": "Start and end frame of data to be processed.",
    },
    "current_media_io_dataset": {
        "name": "Media IO Dataset",
        "description": "Dataset for all media I/O operations",
    },
    "current_operator_progress": {
        "name": "Current Operator Progress",
        "description": "",
    },
    "current_log_entry_hub": {"name": "Current Log Entry Hub", "description": ""},
}
