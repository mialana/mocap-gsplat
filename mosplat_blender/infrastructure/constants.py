"""
constants with maintained minimal imports.
this is reserved both for globals like `_MISSING_`, `ADDON_ID`
and literal structured data (i.e. the below styling literals for the STDOUT logger)
that are not (yet) exposed through Blender add-on preferences or properties.

having this centralized location for these definitions avoids literals that can be easily
misspelled or renamed in one location and not in another.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

_MISSING_: Any = object()  # sentinel variable

_TIMER_INTERVAL_: Final[float] = 0.1
_TIMEOUT_LAZY_: Final[float] = 10.0  # amount of time to wait before timeout
_TIMEOUT_IMMEDIATE_: Final[float] = 0.5

# for pretty logs!
COLORED_FORMATTER_FIELD_STYLES = {
    "asctime": {"faint": True, "underline": True},
    "dirname": {"color": 172, "faint": True},
    "filename": {"color": 184, "faint": True},
    "classname": {"color": 118, "faint": True},
    "funcName": {"color": 117, "faint": True},
    "lineno": {"color": 105, "faint": True},
}

COLORED_FORMATTER_LEVEL_STYLES = {
    "debug": {"color": "magenta"},
    "info": {"color": "green"},
    "warning": {
        "color": "yellow",
        "bold": True,
    },
    "error": {
        "color": "white",
        "bold": True,
        "background": "red",
    },
}

PER_FRAME_DIRNAME: Final[str] = "frame_{:04d}"

MAX_LOG_ENTRIES_STORED: Final[int] = 24
DEFAULT_LOG_ENTRY_ROWS: Final[int] = 8

# path location of the shipped preprocess script
DEFAULT_PREPROCESS_MEDIA_SCRIPT: Final[str] = str(
    Path(__file__).resolve().parent.parent / "bin" / "fix_mocap_video_rotations.py"
)
# target function in script
PREPROCESS_MEDIA_SCRIPT_FUNCTION_NAME: Final[str] = "preprocess"

# path location of hf model download via subprocess script
DOWNLOAD_HF_WITH_PROGRESS_SCRIPT: Final[Path] = (
    Path(__file__).resolve().parent.parent / "lib" / "download_hf_with_progress.py"
)
