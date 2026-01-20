from __future__ import annotations

from typing import Dict, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class PreprocessScriptApplication:
    script_path: str
    application_time: float = -1.0

    @staticmethod
    def now(script_path: str) -> PreprocessScriptApplication:
        return PreprocessScriptApplication(
            script_path=script_path,
            application_time=datetime.now().timestamp(),
        )


@dataclass(frozen=True)
class ProcessedFrameRange:
    start_frame: int
    end_frame: int
    applied_preprocess_scripts: List[PreprocessScriptApplication] = field(
        default_factory=list
    )

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)


@dataclass
class MediaProcessStatus:
    filepath: str
    is_valid: bool = False
    frame_count: int = -1
    message: str = ""
    mod_time: float = -1.0
    file_size: int = -1

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)


@dataclass
class MediaIOMetadata:
    base_directory: str
    do_media_durations_all_match: bool = False
    collective_media_frame_count: int = -1
    media_process_statuses: List[MediaProcessStatus] = field(default_factory=list)
    processed_frame_ranges: List[ProcessedFrameRange] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict):
        instance = cls(**d)
        instance.media_process_statuses = [
            MediaProcessStatus.from_dict(m) for m in instance.media_process_statuses
        ]
