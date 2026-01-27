"""
constants with maintained minimal imports.
this is reserved both for globals like `_MISSING_`, `ADDON_ID`
and literal structured data (i.e. the below styling literals for the STDOUT logger)
that are not (yet) exposed through Blender add-on preferences or properties.

having this centralized location for these definitions avoids literals that can be easily
misspelled or renamed in one location and not in another.
"""

from __future__ import annotations

from typing import Any, Final, TYPE_CHECKING, TypeAlias
from pathlib import Path
from string import capwords
import tempfile

_MISSING_: Any = object()  # sentinel variable

_TIMER_INTERVAL_: Final[float] = 0.1

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
        "color": "red",
        "bold": True,
    },
}

"""
this is the `bl_idname` that blender expects our `AddonPreferences` to have.
i.e. even though our addon id is `mosplat_blender`, the id would be the evaluated
runtime package, which includes the extension repository and the "bl_ext" prefix.
so if this addon is in the `user_default` repository, the id is expected to be:
`bl_ext.user_default.mosplat_blender`.
"""
ADDON_PREFERENCES_ID: Final[str] = (
    __package__.rpartition(".")[
        0
    ]  # remove last part of `__package__` since this file is in a subdirectory
    if __package__
    else Path(__file__).resolve().parent.parent.name
)  # current package is one level down from the one blender expects

ADDON_BASE_ID: Final[str] = ADDON_PREFERENCES_ID.rpartition(".")[-1]

ADDON_HUMAN_READABLE: Final[str] = capwords(ADDON_BASE_ID.replace("_", " "))
ADDON_SHORTNAME: Final[str] = ADDON_BASE_ID.partition("_")[0]

ADDON_TEMP_DIRPATH: Final[Path] = Path(tempfile.mkdtemp(prefix=ADDON_PREFERENCES_ID))

"""
the name of the pointer to `Mosplat_PG_Global` that will be placed on the 
`bpy.context.scene` object for convenient access in operators, panels, etc.
"""
ADDON_PROPERTIES_ATTRIBNAME: Final[str] = f"{ADDON_SHORTNAME}_props"

OPERATOR_ID_PREFIX: Final[str] = f"{ADDON_SHORTNAME}."
PANEL_ID_PREFIX: Final[str] = f"{ADDON_SHORTNAME.upper()}_PT_"

# static typecheck-only abstraction
if TYPE_CHECKING:
    from _typeshed import DataclassInstance
else:
    DataclassInstance: TypeAlias = Any

# path location of the shipped preprocess script
DEFAULT_PREPROCESS_MEDIA_SCRIPT_FILE: Final[str] = str(
    Path(__file__)
    .resolve()
    .parent.parent.joinpath("bin")
    .joinpath("fix_mocap_video_rotations.py")
)

MEDIA_IO_DATASET_JSON_FILENAME: Final[str] = f"{ADDON_SHORTNAME}_data.json"

PER_FRAME_DIRNAME = "frame_{:04d}"
RAW_FRAME_DIRNAME = "raw"


DOWNLOAD_HF_WITH_PROGRESS_SCRIPT_PATH: Final[Path] = (
    Path(__file__)
    .resolve()
    .parent.parent.joinpath("bin")
    .joinpath("download_hf_with_progress.py")
)
