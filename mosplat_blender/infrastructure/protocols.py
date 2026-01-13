"""
protocol classes for typing purposes.
"""

from typing import Protocol


class SupportsMosplat_AP_Global(Protocol):
    """Prevents escape of blender concepts into logging interface."""

    cache_dir: str
    json_log_subdir: str

    json_log_filename_format: str
    json_log_format: str
    json_date_log_format: str

    stdout_log_format: str
    stdout_date_log_format: str
