from __future__ import annotations

from pathlib import Path
from typing import Dict, List, cast, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


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
    base_directory: str
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

    def from_JSON(self, src_path: Path) -> bool:
        if src_path.exists():
            try:
                with src_path.open("r", encoding="utf-8") as f:
                    data: Dict = json.load(f)

                self.from_dict(data)
                return True
            except (TypeError, json.JSONDecodeError):
                src_path.unlink()  # delete the corrupted JSON

        return False


@dataclass
class GlobalData:
    current_media_dir: str = str(Path.home())
    current_frame_range: Tuple[int, int] = field(default_factory=tuple[0, 6])
    current_media_io_metadata: MediaIOMetadata = field(
        default_factory=lambda: MediaIOMetadata(base_directory=str(Path.home()))
    )
