"""
protocol classes for static-typing purposes.
many exist to prevent escape of blender types into .infrastructure
"""

from typing import Protocol, Any


class SupportsMosplat_AP_Global(Protocol):

    cache_dir: str
    json_log_subdir: str

    json_log_filename_format: str
    json_log_format: str
    json_date_log_format: str

    stdout_log_format: str
    stdout_date_log_format: str


class SupportsBpyAddon(Protocol):
    preferences: Any


class SupportsBpyAddons(Protocol):
    def __getitem__(self, key: str) -> SupportsBpyAddon: ...


class SupportsBpyPreferences(Protocol):
    addons: SupportsBpyAddons


class SupportsBpyContext(Protocol):
    preferences: SupportsBpyPreferences
