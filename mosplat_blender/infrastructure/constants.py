from pathlib import Path
from typing import Any

# to be converted to platform-dependent path
ADDON_CACHE_SUBDIR: Path = Path(".cache/mosplat_blender/")

STDOUT_DATE_LOG_FORMAT = "%I:%M:%S %p"
# custom levelletter and dirname logrecord attributes (handled in `MosplatStreamFormatter` class)
STDOUT_LOG_FORMAT = "[%(levelletter)s][%(asctime)s][%(dirname)s::%(filename)s::%(basename)s::%(funcName)s:%(lineno)s] %(message)s"
JSON_DATE_LOG_FORMAT = "%Y-%m-%d %H:%M:%S"
JSON_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(filename)s %(funcName)s %(thread)d %(message)s"

JSON_LOG_OUTSUBDIR: Path = ADDON_CACHE_SUBDIR.joinpath("log")
JSON_LOG_OUTFILE = "mosplat_%Y-%m-%d_%H-%M-%S.log"

# for pretty logs!
COLORED_FORMATTER_FIELD_STYLES = {
    "asctime": {"faint": True, "underline": True},
    "dirname": {"color": 172, "faint": True},
    "filename": {"color": 184, "faint": True},
    "basename": {"color": 118, "faint": True},
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
        "color": "red",
        "bold": True,
    },
}

_MISSING_: Any = object()  # sentinel variable
