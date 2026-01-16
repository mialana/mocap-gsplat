"""
constants with maintained minimal imports.
this is reserved both for globals like `_MISSING_`, `ADDON_ID`
and literal structured data (i.e. the below styling literals for the STDOUT logger)
that are not (yet) exposed through Blender add-on preferences or properties.

having this centralized location for these definitions avoids literals that can be easily
misspelled or renamed in one location and not in another.
"""

from __future__ import annotations

from typing import Any, Final
from pathlib import Path
from enum import StrEnum, auto
from string import capwords

_MISSING_: Any = object()  # sentinel variable

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

# path location of the shipped preprocess script
DEFAULT_PREPROCESS_MEDIA_SCRIPT_FILE = str(
    Path(__file__)
    .resolve()
    .parent.parent.joinpath("bin")
    .joinpath("fix_mocap_video_rotations.py")
)

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
ADDON_SHORTNAME: Final[str] = ADDON_HUMAN_READABLE.partition(" ")[0]

"""
the name of the pointer to `Mosplat_PG_Global` that will be placed on the 
`bpy.context.scene` object for convenient access in operators, panels, etc.
"""
ADDON_PROPERTIES_ATTRIBNAME: Final[str] = f"{ADDON_SHORTNAME.lower()}_props"

OPERATOR_ID_PREFIX: Final[str] = f"{ADDON_SHORTNAME.lower()}."
PANEL_ID_PREFIX: Final[str] = f"{ADDON_SHORTNAME.upper()}_PT_"

"""Enum Convenience Classes"""


class OperatorIDEnum(StrEnum):
    @staticmethod
    def _prefix():
        return OPERATOR_ID_PREFIX

    @staticmethod
    def _category():
        return ADDON_SHORTNAME

    @staticmethod
    def _generate_next_value_(name, start, count, last_values) -> str:
        return f"{OPERATOR_ID_PREFIX}{name.lower()}"

    @staticmethod
    def label_factory(member: OperatorIDEnum):
        """
        creates the operator label from the id
        keeping this here so this file can be a one-stop shop for metadata construction
        """
        return capwords(member.value.removeprefix(OPERATOR_ID_PREFIX).replace("_", " "))

    @staticmethod
    def basename_factory(member: OperatorIDEnum):
        return member.value.rpartition(".")[-1]

    INITIALIZE_MODEL = auto()
    RUN_INFERENCE = auto()
    OPEN_ADDON_PREFERENCES = auto()
    PREPARE_MEDIA_DIRECTORY = auto()


class PanelIDEnum(StrEnum):
    @staticmethod
    def _prefix():
        return PANEL_ID_PREFIX

    @staticmethod
    def _category():
        return ADDON_SHORTNAME

    @staticmethod
    def _generate_next_value_(name, start, count, last_values) -> str:
        return f"{PANEL_ID_PREFIX}{name.lower()}"

    @staticmethod
    def label_factory(member: PanelIDEnum):
        """
        creates the panel label from the id
        keeping this here so this file can be a one-stop shop for metadata construction
        """
        return capwords(member.value.removeprefix(PANEL_ID_PREFIX).replace("_", " "))

    MAIN = auto()
    PREPROCESS = auto()
