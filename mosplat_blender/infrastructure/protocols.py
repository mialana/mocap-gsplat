from typing import Protocol, TypeVar, Optional

R = TypeVar("R")


class SupportsRunOnce(Protocol[R]):
    has_run: bool
    result: Optional[R]

    def __call__(self, *args, **kwargs) -> R: ...


class SupportsMosplat_AddonPreferences(Protocol):
    cache_dir: str

    json_log_subdir: str
    json_log_filename_format: str
    json_log_format: str
    json_date_log_format: str

    stdout_log_format: str
    stdout_date_log_format: str
