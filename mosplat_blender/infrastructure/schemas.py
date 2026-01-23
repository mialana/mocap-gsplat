from __future__ import annotations

from pathlib import Path
from typing import Dict, List, cast, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
from enum import StrEnum, auto
from string import capwords

from .constants import OPERATOR_ID_PREFIX, ADDON_SHORTNAME, PANEL_ID_PREFIX


class PollGuardError(RuntimeError):
    """Create a custom `RuntimeError` for errors that were not guarded correctly by `poll`."""

    def __str__(self) -> str:
        return "Something went wrong with `poll`-guard."


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

    @staticmethod
    def run(bpy_ops, member: OperatorIDEnum):
        getattr(getattr(bpy_ops, member._category()), member.basename_factory(member))()

    INITIALIZE_MODEL = auto()
    RUN_INFERENCE = auto()
    OPEN_ADDON_PREFERENCES = auto()
    CHECK_MEDIA_FRAME_COUNTS = auto()


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


@dataclass(frozen=True)
class PreprocessScriptApplication:
    script_path: str
    application_time: float

    @staticmethod
    def now(script_path: str) -> PreprocessScriptApplication:
        return PreprocessScriptApplication(
            script_path=script_path,
            application_time=datetime.now().timestamp(),
        )

    @classmethod
    def from_dict(cls, d: Dict) -> PreprocessScriptApplication:
        return cls(**d)


@dataclass
class ProcessedFrameRange:
    start_frame: int
    end_frame: int
    applied_preprocess_scripts: List[PreprocessScriptApplication] = field(
        default_factory=list
    )

    @classmethod
    def from_dict(cls, d: Dict) -> ProcessedFrameRange:
        instance = cls(**d)
        instance.applied_preprocess_scripts = [
            PreprocessScriptApplication.from_dict(cast(Dict, a))
            for a in instance.applied_preprocess_scripts
        ]
        return instance


@dataclass
class MediaProcessStatus:
    filepath: str
    is_valid: bool = False
    frame_count: int = -1
    message: str = ""
    mod_time: float = -1.0
    file_size: int = -1

    @classmethod
    def from_dict(cls, d: Dict) -> MediaProcessStatus:
        return cls(**d)


@dataclass
class MediaIOMetadata:
    base_directory: str = str(Path.home())
    do_media_durations_all_match: bool = False
    collective_media_frame_count: int = -1
    media_process_statuses: List[MediaProcessStatus] = field(default_factory=list)
    processed_frame_ranges: List[ProcessedFrameRange] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict) -> MediaIOMetadata:
        instance = cls(**d)
        instance.media_process_statuses = [
            MediaProcessStatus.from_dict(
                cast(Dict, m)
            )  # note that the underlying structure of `m` is still a dict after `cls(**d)`
            for m in instance.media_process_statuses
        ]
        instance.processed_frame_ranges = [
            ProcessedFrameRange.from_dict(cast(Dict, p))
            for p in instance.processed_frame_ranges
        ]
        return instance

    def to_JSON(self, dest_path: Path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        as_dict = asdict(self)

        with dest_path.open("w", encoding="utf-8") as f:
            json.dump(as_dict, f, sort_keys=True, indent=4)

    @classmethod
    def from_JSON(cls, src_path: Path) -> MediaIOMetadata:
        if src_path.exists():
            try:
                with src_path.open("r", encoding="utf-8") as f:
                    data: Dict = json.load(f)

                return cls.from_dict(data)
            except (TypeError, json.JSONDecodeError):
                src_path.unlink()  # delete the corrupted JSON

        return cls()


@dataclass
class GlobalData:
    current_media_dir: str = str(Path.home())
    current_frame_range: Tuple[int, int] = field(default_factory=tuple[0, 6])
    current_media_io_metadata: MediaIOMetadata = field(default_factory=MediaIOMetadata)
    was_restored_from_json: bool = False
