# pyright: reportInvalidTypeForm=false
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Generator, Generic, List, Tuple, Union

from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    IntVectorProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Context, PropertyGroup

from ..infrastructure.constants import RAW_FRAME_DIRNAME
from ..infrastructure.macros import try_access_path
from ..infrastructure.mixins import BlRNAAccessorMixin, D, DataclassInteropMixin
from ..infrastructure.protocols import SupportsCollectionProperty
from ..infrastructure.schemas import (
    AppliedPreprocessScript,
    BlenderEnumItem,
    GlobalData,
    LogEntry,
    LogEntryHub,
    LogEntryLevelEnum,
    MediaFileStatus,
    MediaIODataset,
    OperatorIDEnum,
    OperatorProgress,
    ProcessedFrameRange,
    UserFacingError,
)
from .checks import (
    check_current_media_dirpath,
    check_data_json_filepath,
    check_data_output_dirpath,
    check_frame_range_npy_filepaths,
    check_frame_range_poll_result,
    check_media_files,
)
from .meta.properties_meta import (
    MOSPLAT_PG_APPLIEDPREPROCESSSCRIPT_META,
    MOSPLAT_PG_GLOBAL_META,
    MOSPLAT_PG_LOGENTRY_META,
    MOSPLAT_PG_LOGENTRYHUB_META,
    MOSPLAT_PG_MEDIAFILESTATUS_META,
    MOSPLAT_PG_MEDIAIODATASET_META,
    MOSPLAT_PG_OPERATORPROGRESS_META,
    MOSPLAT_PG_PROCESSEDFRAMERANGE_META,
    Mosplat_PG_AppliedPreprocessScript_Meta,
    Mosplat_PG_Global_Meta,
    Mosplat_PG_LogEntry_Meta,
    Mosplat_PG_LogEntryHub_Meta,
    Mosplat_PG_MediaFileStatus_Meta,
    Mosplat_PG_MediaIODataset_Meta,
    Mosplat_PG_OperatorProgress_Meta,
    Mosplat_PG_ProcessedFrameRange_Meta,
)

if TYPE_CHECKING:
    from .preferences import Mosplat_AP_Global

LogEntryLevelEnumItems: Final[List[BlenderEnumItem]] = [
    member.to_blender_enum_item() for member in LogEntryLevelEnum
]


def update_current_media_dir(self: Mosplat_PG_Global, context: Context):
    OperatorIDEnum.run(OperatorIDEnum.VALIDATE_MEDIA_FILE_STATUSES, "INVOKE_DEFAULT")

    self.logger.info(f"'{self.get_prop_name('current_media_dir')}' updated.")


class MosplatPropertyGroupBase(
    Generic[D],
    PropertyGroup,
    BlRNAAccessorMixin,
    DataclassInteropMixin[D],
):
    pass


class Mosplat_PG_AppliedPreprocessScript(
    MosplatPropertyGroupBase[AppliedPreprocessScript]
):
    _meta: Mosplat_PG_AppliedPreprocessScript_Meta = (
        MOSPLAT_PG_APPLIEDPREPROCESSSCRIPT_META
    )
    __dataclass_type__ = AppliedPreprocessScript

    script_path: StringProperty(name="Script Path", subtype="FILE_PATH")
    mod_time: FloatProperty(name="Modification Time", default=-1.0)
    file_size: IntProperty(name="File Size", default=-1)


class Mosplat_PG_ProcessedFrameRange(MosplatPropertyGroupBase[ProcessedFrameRange]):
    _meta: Mosplat_PG_ProcessedFrameRange_Meta = MOSPLAT_PG_PROCESSEDFRAMERANGE_META
    __dataclass_type__ = ProcessedFrameRange

    start_frame: IntProperty(name="Start Frame", default=0, min=0)
    end_frame: IntProperty(name="End Frame", default=0, min=0)
    applied_preprocess_scripts: CollectionProperty(
        name="Applied Preprocess Scripts", type=Mosplat_PG_AppliedPreprocessScript
    )

    @property
    def scripts_accessor(
        self,
    ) -> SupportsCollectionProperty[Mosplat_PG_AppliedPreprocessScript]:
        return self.applied_preprocess_scripts


class Mosplat_PG_MediaFileStatus(MosplatPropertyGroupBase[MediaFileStatus]):
    _meta: Mosplat_PG_MediaFileStatus_Meta = MOSPLAT_PG_MEDIAFILESTATUS_META
    __dataclass_type__ = MediaFileStatus

    filepath: StringProperty(name="Filepath", subtype="FILE_PATH")
    frame_count: IntProperty(name="Frame Count", default=-1)
    width: IntProperty(name="Width", default=-1)
    height: IntProperty(name="Height", default=-1)
    is_valid: BoolProperty(name="Is Valid", default=False)
    mod_time: FloatProperty(name="Modification Time", default=-1.0)
    file_size: IntProperty(name="File Size", default=-1)

    def matches_dataset(
        self, dataset: Union[Mosplat_PG_MediaIODataset, MediaIODataset]
    ) -> Tuple[bool, bool, bool]:
        return (
            self.frame_count == dataset.median_frame_count,
            self.width == dataset.median_width,
            self.height == dataset.median_height,
        )


class Mosplat_PG_MediaIODataset(MosplatPropertyGroupBase[MediaIODataset]):
    _meta: Mosplat_PG_MediaIODataset_Meta = MOSPLAT_PG_MEDIAIODATASET_META
    __dataclass_type__ = MediaIODataset

    base_directory: StringProperty(
        name="Base Directory",
        description="Filepath to directory containing media files being processed.",
        default=str(Path.home()),
        subtype="DIR_PATH",
    )

    is_valid_media_directory: BoolProperty(
        name="Is Valid Media Directory",
        description="Tracks whether the found media in the current media directory all"
        "have matching frame count, width, and height.",
        default=False,
    )

    median_frame_count: IntProperty(
        name="Median Frame Count",
        description="Median frame count for all media files within the selected media directory.",
        default=-1,
    )

    median_width: IntProperty(
        name="Median Width",
        description="Median width for all media files within the selected media directory.",
        default=-1,
    )

    median_height: IntProperty(
        name="Median Height",
        description="Median height for all media files within the selected media directory.",
        default=-1,
    )

    media_file_statuses: CollectionProperty(
        name="Media File Statuses", type=Mosplat_PG_MediaFileStatus
    )

    processed_frame_ranges: CollectionProperty(
        name="Processed Frame Ranges", type=Mosplat_PG_ProcessedFrameRange
    )

    def to_JSON(self, json_filepath):
        self.to_dataclass().to_JSON(json_filepath)

    @property
    def statuses_accessor(
        self,
    ) -> SupportsCollectionProperty[Mosplat_PG_MediaFileStatus]:
        return self.media_file_statuses

    @property
    def ranges_accessor(
        self,
    ) -> SupportsCollectionProperty[Mosplat_PG_ProcessedFrameRange]:
        return self.processed_frame_ranges


class Mosplat_PG_OperatorProgress(MosplatPropertyGroupBase[OperatorProgress]):
    _meta: Mosplat_PG_OperatorProgress_Meta = MOSPLAT_PG_OPERATORPROGRESS_META
    __dataclass_type__ = OperatorProgress

    current: IntProperty(
        name="Progress Current",
        description="Singleton current progress of operators.",
        default=-1,
    )

    total: IntProperty(
        name="Progress Total",
        description="Singleton total progress of operators.",
        default=-1,
    )

    in_use: BoolProperty(
        name="Progress In Use",
        description="Whether any operator is 'using' the progress-related properties.",
        default=False,
    )


class Mosplat_PG_LogEntry(MosplatPropertyGroupBase[LogEntry]):
    _meta: Mosplat_PG_LogEntry_Meta = MOSPLAT_PG_LOGENTRY_META
    __dataclass_type__ = LogEntry

    level: EnumProperty(
        name="Log Entry Level",
        items=LogEntryLevelEnumItems,
        default=LogEntryLevelEnum.INFO.value,
    )
    message: StringProperty(name="Log Entry Message")
    full_message: StringProperty(
        name="Log Entry Full Message",
        description="The property that is displayed in the dynamic tooltip while hovering on the item.",
    )

    @property
    def level_as_enum(self) -> LogEntryLevelEnum:
        return LogEntryLevelEnum(self.level)


class Mosplat_PG_LogEntryHub(MosplatPropertyGroupBase[LogEntryHub]):
    _meta: Mosplat_PG_LogEntryHub_Meta = MOSPLAT_PG_LOGENTRYHUB_META
    __dataclass_type__ = LogEntryHub

    logs: CollectionProperty(name="Log Entries Data", type=Mosplat_PG_LogEntry)
    logs_active_index: IntProperty(name="Log Entries Active Index", default=0)
    logs_level_filter: EnumProperty(
        name="Log Entries Level Filter",
        items=LogEntryLevelEnumItems,
        default=LogEntryLevelEnum.ALL.value,
    )

    @property
    def entries_accessor(self) -> SupportsCollectionProperty[Mosplat_PG_LogEntry]:
        return self.logs

    @property
    def level_filter_as_enum(self) -> LogEntryLevelEnum:
        return LogEntryLevelEnum(self.logs_level_filter)

    @classmethod
    def data_prop_id(cls) -> str:
        return cls.get_prop_id("logs")

    @classmethod
    def active_index_prop_id(cls) -> str:
        return cls.get_prop_id("logs_active_index")

    @classmethod
    def level_filter_prop_id(cls) -> str:
        return cls.get_prop_id("logs_level_filter")


class Mosplat_PG_Global(MosplatPropertyGroupBase[GlobalData]):
    _meta: Mosplat_PG_Global_Meta = MOSPLAT_PG_GLOBAL_META
    __dataclass_type__ = GlobalData

    current_media_dir: StringProperty(
        name="Media Directory",
        description="Filepath to directory containing media files to be processed.",
        default=str(Path.home()),
        subtype="DIR_PATH",
        update=update_current_media_dir,
        options={"SKIP_SAVE"},
    )

    current_frame_range: IntVectorProperty(
        name="Frame Range",
        description="Start and end frame of data to be processed.",
        size=2,
        default=(0, 60),
        min=0,
        options={"SKIP_SAVE"},
    )

    current_media_io_dataset: PointerProperty(
        name="Media IO Dataset",
        description="Dataset for all media I/O operations",
        type=Mosplat_PG_MediaIODataset,
        options={"SKIP_SAVE"},
    )

    current_operator_progress: PointerProperty(
        name="Current Operator Progress",
        type=Mosplat_PG_OperatorProgress,
        options={"SKIP_SAVE"},
    )

    current_log_entry_hub: PointerProperty(
        name="Current Log Entry Hub", type=Mosplat_PG_LogEntryHub, options={"SKIP_SAVE"}
    )

    @property
    def dataset_accessor(self) -> Mosplat_PG_MediaIODataset:
        return self.current_media_io_dataset

    @property
    def progress_accessor(self) -> Mosplat_PG_OperatorProgress:
        return self.current_operator_progress

    @property
    def log_hub_accessor(self) -> Mosplat_PG_LogEntryHub:
        return self.current_log_entry_hub

    @property
    def current_media_dirpath(self) -> Path:
        return check_current_media_dirpath(self)

    def data_output_dirpath(self, prefs: Mosplat_AP_Global) -> Path:
        return check_data_output_dirpath(prefs, self)

    def data_json_filepath(self, prefs: Mosplat_AP_Global) -> Path:
        return check_data_json_filepath(prefs, self)

    def media_files(self, prefs: Mosplat_AP_Global) -> List[Path]:
        return check_media_files(prefs, self)

    def frame_range_poll_result(self, prefs: Mosplat_AP_Global) -> List[str]:
        return check_frame_range_poll_result(prefs, self)

    @property
    def is_valid_media_directory_poll_result(self) -> List[str]:
        dataset = self.dataset_accessor

        result = []
        if not dataset.is_valid_media_directory:
            result.append(
                f"Ensure matching frame count, width, & height"
                "of media in '{dataset.base_directory}'."
            )
        return result

    def frame_range_npy_filepaths(
        self, prefs: Mosplat_AP_Global, id: str
    ) -> List[Path]:
        """
        raises `UserFacingError` if one of the expected frames do not exist.
        if the files are not expected to exist yet, use the generator fn below instead.
        """
        filepaths: List[Path] = []

        for fp in check_frame_range_npy_filepaths(prefs, self, id):
            try:
                try_access_path(fp)  # raises
                filepaths.append(fp)
            except (OSError, PermissionError, FileNotFoundError) as e:
                raise UserFacingError(
                    f"Expected to find an extracted NPY file at '{fp}'."
                    "Re-extract the frame range if necessary.",
                    e,
                )  # convert now to a `UserFacingError`
        return filepaths

    def generate_frame_range_npy_filepaths(
        self, prefs: Mosplat_AP_Global
    ) -> Generator[Path]:
        """wrapper"""
        return check_frame_range_npy_filepaths(prefs, self, RAW_FRAME_DIRNAME)
