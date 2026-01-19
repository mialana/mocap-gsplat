from __future__ import annotations

from pathlib import Path
from typing import (
    final,
    ClassVar,
    Union,
    TypeAlias,
    List,
    Generator,
    Tuple,
)
from dataclasses import dataclass, asdict, field
from datetime import datetime
import json

from ..infrastructure.decorators import no_instantiate
from ..infrastructure.mixins import MosplatLogClassMixin
from ..infrastructure.constants import MOSPLAT_MEDIA_METADATA_FILENAME

StrPath: TypeAlias = Union[str, Path]


@dataclass(frozen=True)
class MosplatAppliedPreprocessScript:
    script_path: str
    date_last_applied: str

    @staticmethod
    def now(script_path: str) -> MosplatAppliedPreprocessScript:
        return MosplatAppliedPreprocessScript(
            script_path=script_path,
            date_last_applied=datetime.now().isoformat(),
        )


@dataclass(frozen=True)
class MosplatProcessedFrameRange:
    start_frame: int
    end_frame: int
    applied_preprocess_scripts: List[MosplatAppliedPreprocessScript] = field(
        default_factory=list
    )


@dataclass
class MediaProcessEvent:
    filepath: Path
    ok: bool = False
    frame_count: int = -1
    message: str = ""


@dataclass
class MosplatMediaMetadata:
    base_directory: str
    is_valid: bool = False
    collective_frame_count: int = -1
    media_files: List[str] = field(default_factory=list)
    processed_frame_ranges: List[MosplatProcessedFrameRange] = field(
        default_factory=list
    )

    def toJSON(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        as_dict = asdict(self)

        with path.open("w", encoding="utf-8") as f:
            json.dump(as_dict, f, sort_keys=True, indent=4)

    def fromJSON(self, path: Path):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.__init__(**data)

    def try_normalize_frame_ranges(self):
        """combine overlapping frame ranges if they have had the same preprocess scripts applied to them"""

    def apply_media_event(self, event: MediaProcessEvent):
        if not event.ok:
            self.is_valid = False
            return

        if self.collective_frame_count == -1:
            self.collective_frame_count = event.frame_count

        self.media_files.append(str(event.filepath))


@final
@no_instantiate
class MosplatMediaIOInterface(MosplatLogClassMixin):
    initialized: ClassVar[bool] = False

    @classmethod
    def initialize(cls, base_directory: Path, data_output_dir: Path):
        cls.metadata = MosplatMediaMetadata(base_directory=str(base_directory))
        cls.data_output_dir = data_output_dir
        cls._find_or_create_metadata_json()

        cls.initialized = True

    @classmethod
    def _find_or_create_metadata_json(cls):
        json_filepath = cls.data_output_dir.joinpath(MOSPLAT_MEDIA_METADATA_FILENAME)

    @classmethod
    def process_media_file(
        cls, filepath: Path
    ) -> Generator[MediaProcessEvent, None, None]:
        if not cls.initialized:
            raise RuntimeError(f"`{cls.__qualname__}` not initialized.")

        event = MediaProcessEvent(filepath=filepath)

        try:
            event.frame_count, event.message = cls._get_media_frame_count(filepath)
            event.ok = True
        except RuntimeError as e:
            event.ok = False
            event.message = str(e)

        yield event

    @staticmethod
    def _get_media_frame_count(filepath: Path) -> Tuple[int, str]:
        """use opencv to get the frame count of media"""
        import cv2

        def _cleanup(method: str):
            cap.release()
            return (
                frame_count,
                f"Read video file '{filepath}' with the duration '{frame_count}' frames ({method}).",
            )

        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open media file: {filepath}")

        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)  # seek to end
        duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps > 0 and duration_ms > 0:
            frame_count = int(round((duration_ms / 1000.0) * fps))
            if frame_count > 0:
                return _cleanup("fps + duration metadata")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if 0 < frame_count < 2**32 - 1:
            return _cleanup("frame count metadata")

        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.0)  # return seek to start

        frame_count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            frame_count += 1

        return _cleanup("manual")

    @classmethod
    def apply_preprocess_script(cls, script_path: Path) -> bool:
        return False

    @classmethod
    def extract_frame_range(cls, frame_range):
        pass
