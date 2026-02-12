"""Enum Convenience Classes"""

from __future__ import annotations

from enum import StrEnum, auto
from string import capwords

from mosplat_blender.infrastructure.schemas import AddonMeta


class OperatorIDEnum(StrEnum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values) -> str:
        return f"{AddonMeta().global_operator_prefix}{name.lower()}"  # evaluated first

    @staticmethod
    def _prefix() -> str:
        return AddonMeta().global_operator_prefix

    @staticmethod
    def _category() -> str:
        return AddonMeta().shortname

    @staticmethod
    def label_factory(member: OperatorIDEnum) -> str:
        """
        creates the operator label from the id
        keeping this here so this file can be a one-stop shop for metadata construction
        """
        return capwords(member.value.removeprefix(member._prefix()).replace("_", " "))

    @staticmethod
    def basename_factory(member: OperatorIDEnum) -> str:
        return member.value.rpartition(".")[-1]

    @staticmethod
    def run(member: OperatorIDEnum, *args, **kwargs):
        from bpy import ops  # local import

        getattr(getattr(ops, member._category()), member.basename_factory(member))(
            *args, **kwargs
        )

    INITIALIZE_MODEL = auto()
    OPEN_ADDON_PREFERENCES = auto()
    VALIDATE_FILE_STATUSES = auto()
    EXTRACT_FRAME_RANGE = auto()
    RUN_PREPROCESS_SCRIPT = auto()
    RUN_INFERENCE = auto()
    INSTALL_POINTCLOUD_PREVIEW = auto()


class PanelIDEnum(StrEnum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values) -> str:
        return f"{AddonMeta().global_panel_prefix}{name.lower()}"

    @staticmethod
    def _prefix():
        return AddonMeta().global_panel_prefix

    @staticmethod
    def _category():
        return AddonMeta().shortname.capitalize()

    @staticmethod
    def label_factory(member: PanelIDEnum):
        """
        creates the panel label from the id
        keeping this here so this file can be a one-stop shop for metadata construction
        """
        return capwords(member.value.removeprefix(member._prefix()).replace("_", " "))

    MAIN = auto()
    PREPROCESS = auto()
    DATA_INFERENCE = auto()
    LOG_ENTRIES = auto()


class UIListIDEnum(StrEnum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values) -> str:
        return f"{AddonMeta().global_ui_list_prefix}{name.lower()}"

    @staticmethod
    def _prefix():
        return AddonMeta().global_ui_list_prefix

    LOG_ENTRIES = auto()
