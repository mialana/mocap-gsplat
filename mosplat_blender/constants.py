from pathlib import Path

LOGGER_NAME = "mosplat_blender"

# to be converted to platform-dependent path
ADDON_CACHE_SUBDIR: Path = Path(".cache/mosplat_blender/")

STDOUT_DATE_LOG_FORMAT = "%I:%M:%S %p"
# custom levelletter and dirname logrecord attributes (handled in `MosplatStreamFormatter` class)
STDOUT_LOG_FORMAT = "[%(levelletter)s][%(asctime)s][%(dirname)s::%(filename)s::%(funcName)s:%(lineno)s] %(message)s"
JSON_DATE_LOG_FORMAT = "%Y-%m-%d %H:%M:%S"
JSON_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(filename)s %(funcName)s %(thread)d %(message)s"

JSON_LOG_OUTSUBDIR: Path = ADDON_CACHE_SUBDIR.joinpath("log")
JSON_LOG_OUTFILE = "mosplat_%Y-%m-%d_%H-%M-%S.log"

# for pretty logs!
COLORED_FORMATTER_FIELD_STYLES = {
    "asctime": {"color": "cyan"},
    "dirname": {"color": 66},
    "filename": {"color": 179},
    "funcName": {"color": 131},
    "lineno": {"color": 103},
}

COLORED_FORMATTER_LEVEL_STYLES = {
    "info": {"color": "green"},
    "debug": {"color": "yellow"},
}
