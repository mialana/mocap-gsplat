# pyright: reportInvalidTypeForm=false
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Generic, List, Tuple, TypeAlias

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

from core.checks import (
    check_frame_range_poll_result,
    check_media_directory,
    check_media_files,
    check_media_io_metadata_filepath,
    check_media_output_dir,
)
from core.meta.properties_meta import (
    MOSPLAT_PG_APPLIEDPREPROCESSSCRIPT_META,
    MOSPLAT_PG_GLOBAL_META,
    MOSPLAT_PG_LOGENTRY_META,
    MOSPLAT_PG_LOGENTRYHUB_META,
    MOSPLAT_PG_MEDIAFILESTATUS_META,
    MOSPLAT_PG_MEDIAIOMETADATA_META,
    MOSPLAT_PG_OPERATORPROGRESS_META,
    MOSPLAT_PG_PROCESSEDFRAMERANGE_META,
    MOSPLAT_PG_VGGTMODELOPTIONS_META,
    Mosplat_PG_AppliedPreprocessScript_Meta,
    Mosplat_PG_Global_Meta,
    Mosplat_PG_LogEntry_Meta,
    Mosplat_PG_LogEntryHub_Meta,
    Mosplat_PG_MediaFileStatus_Meta,
    Mosplat_PG_MediaIOMetadata_Meta,
    Mosplat_PG_OperatorProgress_Meta,
    Mosplat_PG_ProcessedFrameRange_Meta,
    Mosplat_PG_VGGTModelOptions_Meta,
)
from infrastructure.constants import PER_FRAME_DIRNAME
from infrastructure.identifiers import OperatorIDEnum
from infrastructure.mixins import D, DataclassInteropMixin, EnforceAttributesMixin
from infrastructure.protocols import SupportsCollectionProperty
from infrastructure.schemas import (
    AppliedPreprocessScript,
    BlenderEnumItem,
    LogEntryLevelEnum,
    MediaFileStatus,
    MediaIOMetadata,
    ModelInferenceMode,
    ProcessedFrameRange,
    SavedTensorFileName,
    TensorFileFormatLookup,
    VGGTModelOptions,
)

if TYPE_CHECKING:
    from core.preferences import Mosplat_AP_Global

LogEntryLevelEnumItems: Final[List[BlenderEnumItem]] = [
    member.to_blender_enum_item() for member in LogEntryLevelEnum
]

ModelInferenceModeEnumItems: Final[List[BlenderEnumItem]] = [
    member.to_blender_enum_item() for member in ModelInferenceMode
]

FrameRangeTuple: TypeAlias = Tuple[int, int]


def update_frame_range(self: Mosplat_PG_Global, context: Context):
    data = self.metadata_accessor.to_dataclass()

    start, end = self.frame_range_
    self.was_frame_range_extracted = bool(data.query_frame_range(start, end - 1))

    self.logger.info(f"Frame range updated to '{start}-{end}'.")


def update_media_directory(self: Mosplat_PG_Global, context: Context):
    OperatorIDEnum.run(OperatorIDEnum.VALIDATE_FILE_STATUSES, "INVOKE_DEFAULT")

    self.logger.info(f"'{self._meta.media_directory.name}' updated.")

    update_frame_range(self, context)


class MosplatPropertyGroupBase(
    Generic[D],
    PropertyGroup,
    EnforceAttributesMixin,
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

    file_path: StringProperty(name="File Path", subtype="FILE_PATH")
    mod_time: FloatProperty(name="Modification Time", default=-1.0)
    file_size: IntProperty(name="File Size", default=-1)


class Mosplat_PG_ProcessedFrameRange(MosplatPropertyGroupBase[ProcessedFrameRange]):
    _meta: Mosplat_PG_ProcessedFrameRange_Meta = MOSPLAT_PG_PROCESSEDFRAMERANGE_META
    __dataclass_type__ = ProcessedFrameRange

    start_frame: IntProperty(name="Start Frame", min=0)
    end_frame: IntProperty(name="End Frame", min=0)
    applied_preprocess_script: PointerProperty(
        name="Applied Preprocess Script", type=Mosplat_PG_AppliedPreprocessScript
    )

    @property
    def script_accessor(self) -> Mosplat_PG_AppliedPreprocessScript:
        return self.applied_preprocess_script


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


class Mosplat_PG_MediaIOMetadata(MosplatPropertyGroupBase[MediaIOMetadata]):
    _meta: Mosplat_PG_MediaIOMetadata_Meta = MOSPLAT_PG_MEDIAIOMETADATA_META
    __dataclass_type__ = MediaIOMetadata

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


class Mosplat_PG_OperatorProgress(MosplatPropertyGroupBase):
    _meta: Mosplat_PG_OperatorProgress_Meta = MOSPLAT_PG_OPERATORPROGRESS_META
    __dataclass_type__ = None

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


class Mosplat_PG_LogEntry(MosplatPropertyGroupBase):
    _meta: Mosplat_PG_LogEntry_Meta = MOSPLAT_PG_LOGENTRY_META
    __dataclass_type__ = None

    level: EnumProperty(
        name="Log Entry Level",
        items=LogEntryLevelEnumItems,
        default=LogEntryLevelEnum.INFO.value,
    )
    message: StringProperty(name="Log Entry Message")
    session_index: IntProperty(
        name="Log Session Index",
        description="The self-stored index represented as a monotonic increasing index since the session start.",
        default=-1,
    )
    full_message: StringProperty(
        name="Log Entry Full Message",
        description="The property that is displayed in the dynamic tooltip while hovering on the log item.",
    )


class Mosplat_PG_LogEntryHub(MosplatPropertyGroupBase):
    _meta: Mosplat_PG_LogEntryHub_Meta = MOSPLAT_PG_LOGENTRYHUB_META
    __dataclass_type__ = None

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


class Mosplat_PG_VGGTModelOptions(MosplatPropertyGroupBase):
    _meta: Mosplat_PG_VGGTModelOptions_Meta = MOSPLAT_PG_VGGTMODELOPTIONS_META
    __dataclass_type__ = VGGTModelOptions

    inference_mode: EnumProperty(
        name="Inference Mode",
        items=ModelInferenceModeEnumItems,
        default=ModelInferenceMode.POINT_MAP.value,
    )
    confidence_percentile: FloatProperty(name="Confidence Percentile", default=95.0)
    enable_black_mask: BoolProperty(name="Enable Black Mask", default=True)
    enable_white_mask: BoolProperty(name="Enable White Mask", default=False)


class Mosplat_PG_Global(MosplatPropertyGroupBase):
    _meta: Mosplat_PG_Global_Meta = MOSPLAT_PG_GLOBAL_META
    __dataclass_type__ = None

    media_directory: StringProperty(
        name="Media Directory",
        description="Filepath to directory containing media files to be processed.",
        default=str(Path.home() / "Desktop" / "caroline_shot"),
        subtype="DIR_PATH",
        update=update_media_directory,
        options={"SKIP_SAVE"},
    )

    frame_range: IntVectorProperty(
        name="Frame Range",
        description="Start and end frame of data to be processed.",
        size=2,
        default=(0, 12),
        min=0,
        options={"SKIP_SAVE"},
        update=update_frame_range,
    )

    was_frame_range_extracted: BoolProperty(
        name="Was Frame Range Extracted",
        description="Tracks whether the currently selected frame range extracted already.",
        default=False,
        options={"SKIP_SAVE"},
    )

    operator_progress: PointerProperty(
        name="Operator Progress",
        type=Mosplat_PG_OperatorProgress,
        options={"SKIP_SAVE"},
    )

    log_entry_hub: PointerProperty(
        name="Log Entry Hub", type=Mosplat_PG_LogEntryHub, options={"SKIP_SAVE"}
    )

    vggt_model_options: PointerProperty(
        name="VGGT Model Options",
        type=Mosplat_PG_VGGTModelOptions,
        options={"SKIP_SAVE"},
    )

    media_io_metadata: PointerProperty(
        name="Media IO Metadata",
        description="Metadata for all media I/O operations",
        type=Mosplat_PG_MediaIOMetadata,
        options={"SKIP_SAVE"},
    )

    @property
    def metadata_accessor(self) -> Mosplat_PG_MediaIOMetadata:
        return self.media_io_metadata

    @property
    def progress_accessor(self) -> Mosplat_PG_OperatorProgress:
        return self.operator_progress

    @property
    def options_accessor(self) -> Mosplat_PG_VGGTModelOptions:
        return self.vggt_model_options

    @property
    def log_hub_accessor(self) -> Mosplat_PG_LogEntryHub:
        return self.log_entry_hub

    @property
    def media_directory_(self) -> Path:
        return check_media_directory(self)

    @property
    def frame_range_(self) -> FrameRangeTuple:
        return tuple(self.frame_range)

    def media_data_output_dir_(self, prefs: Mosplat_AP_Global) -> Path:
        return check_media_output_dir(prefs, self)

    def media_io_metadata_filepath(self, prefs: Mosplat_AP_Global) -> Path:
        return check_media_io_metadata_filepath(prefs, self)

    def media_files(self, prefs: Mosplat_AP_Global) -> List[Path]:
        return check_media_files(prefs, self)

    def frame_range_poll_result(self, prefs: Mosplat_AP_Global) -> List[str]:
        return check_frame_range_poll_result(prefs, self)

    @property
    def is_valid_media_directory_poll_result(self) -> List[str]:
        data = self.metadata_accessor

        result = []
        if not data.is_valid_media_directory:
            result.append(
                f"Ensure matching frame count, width, & height of media in '{data.base_directory}'."
            )
        return result

    def generate_safetensor_filepath_formatters(
        self, prefs: Mosplat_AP_Global, names: List[SavedTensorFileName]
    ) -> TensorFileFormatLookup:
        data_dir = check_media_output_dir(prefs, self)
        return {
            name: str(data_dir / PER_FRAME_DIRNAME / f"{name}.safetensors")
            for name in names
        }
