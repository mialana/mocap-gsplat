# auto-generated in build: 2026-02-01 13:57:20.305518


from __future__ import annotations

Mosplat_PG_AppliedPreprocessScriptMeta = {
    "script_path": {"name": "Script Path", "description": ""},
    "mod_time": {"name": "Modification Time", "description": ""},
    "file_size": {"name": "File Size", "description": ""},
}

Mosplat_PG_ProcessedFrameRangeMeta = {
    "start_frame": {"name": "Start Frame", "description": ""},
    "end_frame": {"name": "End Frame", "description": ""},
    "applied_preprocess_scripts": {
        "name": "Applied Preprocess Scripts",
        "description": "",
    },
}

Mosplat_PG_MediaFileStatusMeta = {
    "filepath": {"name": "Filepath", "description": ""},
    "frame_count": {"name": "Frame Count", "description": ""},
    "width": {"name": "Width", "description": ""},
    "height": {"name": "Height", "description": ""},
    "is_valid": {"name": "Is Valid", "description": ""},
    "mod_time": {"name": "Modification Time", "description": ""},
    "file_size": {"name": "File Size", "description": ""},
}

Mosplat_PG_MediaIODatasetMeta = {
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

Mosplat_PG_OperatorProgressMeta = {
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

Mosplat_PG_LogEntryMeta = {
    "level": {"name": "Log Entry Level", "description": ""},
    "message": {"name": "Log Entry Message", "description": ""},
    "full_message": {
        "name": "Log Entry Full Message",
        "description": "The property that is displayed in the dynamic tooltip while hovering on the item.",
    },
}

Mosplat_PG_LogEntryHubMeta = {
    "logs": {"name": "Log Entries Data", "description": ""},
    "logs_active_index": {"name": "Log Entries Active Index", "description": ""},
    "logs_level_filter": {"name": "Log Entries Level Filter", "description": ""},
}

Mosplat_PG_GlobalMeta = {
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
