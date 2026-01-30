"""
defines native errors and dataclasses.
dataclass member methods should raise standard library error types.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Dict,
    List,
    cast,
    Tuple,
    TYPE_CHECKING,
    Union,
    Callable,
    Tuple,
    Optional,
    ClassVar,
    Literal,
    Final,
    get_args,
    TypeAlias,
)

from dataclasses import dataclass, field, asdict
import json
from enum import StrEnum, auto
from string import capwords
from abc import ABC

from .constants import OPERATOR_ID_PREFIX, ADDON_SHORTNAME, PANEL_ID_PREFIX
from .macros import (
    append_if_not_equals,
    int_median,
    try_access_path,
)

if TYPE_CHECKING:
    from cv2 import VideoCapture
    from ..core.properties import Mosplat_PG_MediaIODataset


class CustomError(ABC, RuntimeError):
    base_msg: ClassVar[str]

    @classmethod
    def make_msg(
        cls,
        custom_msg: str = "",
        orig: Optional[BaseException] = None,
        show_orig_msg: bool = True,
    ) -> str:
        orig_type = f" ({type(orig).__name__})" if orig else ""
        msg = f": {custom_msg}" if custom_msg else ""
        orig_msg = (
            f"\nORIGINAL ERROR MSG: {str(orig)}" if orig and show_orig_msg else ""
        )
        return f"{cls.base_msg}{orig_type}{msg}{orig_msg}"

    def __init__(
        self,
        custom_msg: str = "",
        orig: Optional[BaseException] = None,
        show_orig_msg: bool = True,
    ):
        super().__init__(self.make_msg(custom_msg, orig, show_orig_msg))


class UserFacingError(CustomError):
    """a custom `RuntimeError` for errors that are user-caused and user-facing (i.e. should be visible to user)."""

    base_msg = "USER ERROR"


class DeveloperError(CustomError):
    """a custom `RuntimeError` for developer logic errors."""

    base_msg = "Developer error (you are doing something wrong)"


class UnexpectedError(CustomError):
    """a custom `RuntimeError` for errors that actually should never occur."""

    base_msg = "Something went wrong"


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
    def run(member: OperatorIDEnum, *args, **kwargs):
        from bpy import ops  # local import

        getattr(getattr(ops, member._category()), member.basename_factory(member))(
            *args, **kwargs
        )

    INITIALIZE_MODEL = auto()
    RUN_INFERENCE = auto()
    OPEN_ADDON_PREFERENCES = auto()
    VALIDATE_MEDIA_FILE_STATUSES = auto()
    EXTRACT_FRAME_RANGE = auto()
    RUN_PREPROCESS_SCRIPT = auto()
    REPORT_GLOBAL_MESSAGES = auto()


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


WmReportItems: TypeAlias = Literal[
    "DEBUG",
    "INFO",
    "OPERATOR",
    "PROPERTY",
    "WARNING",
    "ERROR",
    "ERROR_INVALID_INPUT",
    "ERROR_INVALID_CONTEXT",
    "ERROR_OUT_OF_MEMORY",
]


class WmReportItemsEnum(StrEnum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values) -> str:
        return name.upper()

    DEBUG = auto()
    INFO = auto()
    OPERATOR = auto()
    PROPERTY = auto()
    WARNING = auto()
    ERROR = auto()
    ERROR_INVALID_INPUT = auto()
    ERROR_INVALID_CONTEXT = auto()
    ERROR_OUT_OF_MEMORY = auto()


@dataclass(frozen=True)
class AppliedPreprocessScript:
    script_path: str
    mod_time: float
    file_size: int

    @classmethod
    def from_dict(cls, d: Dict) -> AppliedPreprocessScript:
        return cls(**d)

    @classmethod
    def from_script_path(cls, script_path: str) -> AppliedPreprocessScript:
        script_filepath = Path(script_path)
        stat = try_access_path(script_filepath)

        return cls(
            script_path=script_path, mod_time=stat.st_mtime, file_size=stat.st_size
        )


@dataclass
class ProcessedFrameRange:
    start_frame: int
    end_frame: int
    applied_preprocess_scripts: List[AppliedPreprocessScript] = field(
        default_factory=list
    )

    @classmethod
    def from_dict(cls, d: Dict) -> ProcessedFrameRange:
        instance = cls(**d)
        instance.applied_preprocess_scripts = [
            AppliedPreprocessScript.from_dict(cast(Dict, a))
            for a in instance.applied_preprocess_scripts
        ]
        return instance


@dataclass
class MediaFileStatus:
    filepath: str
    is_valid: bool = False
    frame_count: int = -1
    width: int = -1
    height: int = -1
    mod_time: float = -1.0
    file_size: int = -1

    @classmethod
    def from_dict(cls, d: Dict) -> MediaFileStatus:
        return cls(**d)

    def overwrite(
        self,
        *,
        is_valid: bool,
        frame_count: int,
        width: int,
        height: int,
        mod_time: float,
        file_size: int,
    ):
        self.is_valid = is_valid
        self.frame_count = frame_count
        self.width = width
        self.height = height
        self.mod_time = mod_time
        self.file_size = file_size

    def mark_invalid(self):
        self.is_valid = False
        self.frame_count = -1
        self.width = -1
        self.height = -1
        self.mod_time = -1.0
        self.file_size = -1

    def needs_reextraction(self, dataset) -> bool:
        return (
            not self.is_valid
            or not all(self.matches_dataset(dataset))
            or self.mod_time == -1.0
            or self.file_size == -1
        )

    def matches_dataset(
        self, dataset: Union[MediaIODataset, Mosplat_PG_MediaIODataset]
    ) -> Tuple[bool, bool, bool]:
        return (
            self.frame_count == dataset.median_frame_count,
            self.width == dataset.median_width,
            self.height == dataset.median_height,
        )

    @classmethod
    def as_lookup(cls, statuses: List[MediaFileStatus]) -> Dict[str, MediaFileStatus]:
        return {s.filepath: s for s in statuses}

    def extract_from_filepath(self):
        """use opencv to get width, height, and frame count of media"""
        import cv2

        cap = cv2.VideoCapture(self.filepath)
        if not cap.isOpened():
            self.mark_invalid()
            raise OSError(f"Could not open media file: '{self.filepath}'")

        stat = Path(self.filepath).stat()

        self.overwrite(
            is_valid=True,
            frame_count=self._extract_frame_count(cap),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or -1,  # if returns 0
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or -1,  # if returns 0
            mod_time=stat.st_mtime,
            file_size=stat.st_size,
        )

        cap.release()
        return

    @staticmethod
    def _extract_frame_count(cap: VideoCapture) -> int:
        import cv2

        frame_count = -1
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)  # seek to end
        duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps > 0 and duration_ms > 0:
            estimated = int(round((duration_ms / 1000.0) * fps))
            if estimated > 0:
                frame_count = estimated
        else:
            reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if 0 < reported < 2**32 - 1:
                frame_count = reported
            else:
                cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.0)  # return seek to start

                count = 0
                while True:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    count += 1

                frame_count = count

        return frame_count


@dataclass
class MediaIODataset:
    base_directory: str
    is_valid_media_directory: bool = False
    median_frame_count: int = -1
    median_width: int = -1
    median_height: int = -1
    media_file_statuses: List[MediaFileStatus] = field(default_factory=list)
    processed_frame_ranges: List[ProcessedFrameRange] = field(default_factory=list)

    def load_from_dict(self, d: Dict) -> None:
        """mutates an instance with data from a dict"""
        self.base_directory = d["base_directory"]
        self.is_valid_media_directory = d["is_valid_media_directory"]
        self.median_frame_count = d["median_frame_count"]
        self.median_width = d["median_width"]
        self.median_height = d["median_height"]

        self.media_file_statuses = [
            MediaFileStatus.from_dict(cast(Dict, m))
            for m in d.get("media_file_statuses", [])
        ]

        self.processed_frame_ranges = [
            ProcessedFrameRange.from_dict(cast(Dict, p))
            for p in d.get("processed_frame_ranges", [])
        ]

    @classmethod
    def from_dict(cls, d: Dict) -> MediaIODataset:
        instance = cls(base_directory=d["base_directory"])
        instance.load_from_dict(d)
        return instance

    def to_JSON(self, dest_path: Path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        as_dict = asdict(self)

        with dest_path.open("w", encoding="utf-8") as f:
            json.dump(as_dict, f, sort_keys=True, indent=4)

    def load_from_JSON(self, *, json_path: Path) -> str:
        try:
            try_access_path(json_path)  # do not catch `OSError` and `PermissionError`
        except FileNotFoundError as e:
            e.add_note("Dataclasses fields will stay as is.")
            return str(e)  # the file not existing is fine.
        except (OSError, PermissionError) as e:
            e.add_note(
                "A permission error occured while trying to restore the cached data.\n"
                "Please double-check your machine's file system permissions before continuing."
            )
            raise UserWarning from e  # try to stick with native error classes here

        try:
            with json_path.open("r", encoding="utf-8") as f:
                data: Dict = json.load(f)
            self.load_from_dict(data)
            return f"Cached data successfully loaded from '{json_path}'."

        except (TypeError, json.JSONDecodeError):
            json_path.unlink()  # delete the corrupted cached JSON
            return f"The data cache existed at '{json_path}' but could not be loaded. Deleted the file and will build a new dataset from scratch."

    @classmethod
    def from_JSON(
        cls, *, json_path: Path, base_directory: Path
    ) -> Tuple[MediaIODataset, str]:
        instance = cls(base_directory=str(base_directory))
        load_msg = instance.load_from_JSON(json_path=json_path)
        return (instance, load_msg)

    def synchronize_to_medians(self):
        _frame_counts: List[int] = []
        _widths: List[int] = []
        _heights: List[int] = []
        for status in self.media_file_statuses:
            # do not count in median if the property was not found
            append_if_not_equals(_frame_counts, item=status.frame_count, target=-1)
            append_if_not_equals(_widths, item=status.width, target=-1)
            append_if_not_equals(_heights, item=status.height, target=-1)

        self.median_frame_count = int_median(_frame_counts)
        self.median_width = int_median(_widths)
        self.median_height = int_median(_heights)

    def _accumulate_media_status(self, status: MediaFileStatus) -> None:
        self.media_file_statuses.append(status)
        self.synchronize_to_medians()

        self.is_valid_media_directory = (
            self.is_valid_media_directory
            and status.is_valid
            and self.median_frame_count == status.frame_count
            and self.median_width == status.width
            and self.median_height == status.height
        )

    def status_accumulator(
        self,
    ) -> Tuple[Dict[str, MediaFileStatus], Callable[[MediaFileStatus], None]]:
        """
        sets up state for a new status accumulation.
        returns a lookup table that is created before current statuses are cleared,
        and a callable that will handle adding a new status to the current known statuses.
        """

        status_lookup = MediaFileStatus.as_lookup(self.media_file_statuses)

        # clear the statuses as we've created a lookup table already
        self.media_file_statuses.clear()
        self.is_valid_media_directory = True  # start fresh

        return status_lookup, self._accumulate_media_status


@dataclass
class GlobalData:
    current_media_dir: str = str(Path.home())
    current_frame_range: Tuple[int, int] = field(default_factory=tuple[0, 6])
    current_media_io_dataset: MediaIODataset = field(
        default_factory=lambda: MediaIODataset(base_directory=str(Path.home()))
    )
    operator_progress_current: int = -1
    operator_progress_total: int = -1
    progress_in_use: bool = False
