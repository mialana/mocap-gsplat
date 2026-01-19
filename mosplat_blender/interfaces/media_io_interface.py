from __future__ import annotations

from pathlib import Path
from typing import final, ClassVar, Union, TypeAlias, List, Generator, Tuple, Dict
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

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, Dict):
            raise TypeError("Use this method with dictionary objects.")
        return cls(**d)


@dataclass
class MediaProcessStatus:
    filepath: str
    ok: bool = False
    frame_count: int = -1
    message: str = ""
    mtime: float = -1.0
    size: int = -1

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, Dict):
            raise TypeError("Use this method with dictionary objects.")
        return cls(**d)


@dataclass
class MosplatMediaMetadata:
    base_directory: str
    is_valid: bool = False
    collective_frame_count: int = -1
    media_statuses: List[MediaProcessStatus] = field(default_factory=list)
    processed_frame_ranges: List[MosplatProcessedFrameRange] = field(
        default_factory=list
    )

    def to_JSON(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        as_dict = asdict(self)

        with path.open("w", encoding="utf-8") as f:
            json.dump(as_dict, f, sort_keys=True, indent=4)

    def from_JSON(self, path: Path) -> bool:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    data: Dict = json.load(f)

                self.__init__(**data)

                # restore nested dataclasses that were converted to dictionary objects
                self.media_statuses = [
                    MediaProcessStatus.from_dict(s) for s in self.media_statuses
                ]
                self.processed_frame_ranges = [
                    MosplatProcessedFrameRange.from_dict(s)
                    for s in self.processed_frame_ranges
                ]
                return True
            except (TypeError, json.JSONDecodeError):
                path.unlink()  # delete the corrupted JSON

        return False

    def try_normalize_frame_ranges(self):
        """combine overlapping frame ranges if they have had the same preprocess scripts applied to them"""

    def handle_media_status(self, status: MediaProcessStatus):
        if self.collective_frame_count == -1:
            self.collective_frame_count = status.frame_count

        if status.frame_count != self.collective_frame_count:
            status.ok = False
            status.message = f"Found frame count '{status.frame_count}' for '{status.filepath}' but it does not match the collective frame count of '{self.collective_frame_count}'."

            self.is_valid = False
        else:
            status.ok = True
            self.is_valid = True

        for idx, s in enumerate(self.media_statuses):
            if s.filepath == status.filepath:
                self.media_statuses.pop(idx)

        self.media_statuses.append(status)

    def get_cached_media_status(
        self, filepath: Path
    ) -> Union[MediaProcessStatus, None]:
        fp = str(filepath)
        for status in self.media_statuses:
            if status.filepath == fp:
                stat = filepath.stat()
                if (
                    status.ok
                    and status.frame_count > 0
                    and status.mtime == stat.st_mtime
                    and status.size == stat.st_size
                ):  # ensure the cached status is still valid
                    status.message = f"Loaded cached status video file '{status.filepath}' with the frame count '{status.frame_count}'."
                    return status
        return None


@final
@no_instantiate
class MosplatMediaIOInterface(MosplatLogClassMixin):
    initialized: ClassVar[bool] = False

    @classmethod
    def initialize(cls, base_directory: Path, data_output_dir: Path):
        cls.metadata = MosplatMediaMetadata(base_directory=str(base_directory))
        cls.data_output_dir = data_output_dir
        json_filepath = cls.data_output_dir.joinpath(MOSPLAT_MEDIA_METADATA_FILENAME)

        if cls.metadata.from_JSON(json_filepath):
            cls.logger().info(f"Loaded existing metadata frm '{json_filepath}'.")

        cls.initialized = True

    @classmethod
    def update_metadata_json(cls):
        if not cls.initialized:
            raise RuntimeError(f"`{cls.__qualname__}` not initialized.")

        json_filepath = cls.data_output_dir.joinpath(MOSPLAT_MEDIA_METADATA_FILENAME)
        cls.metadata.to_JSON(json_filepath)

    @classmethod
    def process_media_file(
        cls, filepath: Path
    ) -> Generator[MediaProcessStatus, None, None]:
        if not cls.initialized:
            raise RuntimeError(f"`{cls.__qualname__}` not initialized.")

        status = cls.metadata.get_cached_media_status(filepath)
        if not status:
            # cache could not be used
            status = MediaProcessStatus(filepath=str(filepath))

            try:
                status.frame_count, status.message = cls._get_media_frame_count(
                    filepath
                )
                stat = filepath.stat()
                status.mtime = stat.st_mtime
                status.size = stat.st_size
                cls.metadata.handle_media_status(status)
            except RuntimeError as e:
                status.message = str(e)
                status.ok = False

        yield status

    @staticmethod
    def _get_media_frame_count(filepath: Path) -> Tuple[int, str]:
        """use opencv to get the frame count of media"""
        import cv2

        def _cleanup(method: str):
            cap.release()
            return (
                frame_count,
                f"Read media file '{filepath}' with the frame count '{frame_count}' ({method}).",
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
