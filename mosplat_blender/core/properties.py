# pyright: reportInvalidTypeForm=false
from __future__ import annotations

import bpy
from bpy.types import PropertyGroup, Context
from bpy.props import (
    BoolProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
    CollectionProperty,
    PointerProperty,
    IntVectorProperty,
)

from typing import Generic, TypeVar, TypeAlias, TYPE_CHECKING, Any, List

from pathlib import Path

from .checks import (
    check_media_files,
    check_data_output_dirpath,
    check_data_json_filepath,
    check_current_media_dirpath,
)

from ..infrastructure.mixins import (
    MosplatBlPropertyAccessorMixin,
    MosplatDataclassInteropMixin,
)
from ..infrastructure.protocols import SupportsCollectionProperty
from ..infrastructure.constants import DataclassInstance
from ..infrastructure.schemas import (
    OperatorIDEnum,
    GlobalData,
    MediaIODataset,
    MediaFileStatus,
    ProcessedFrameRange,
    PreprocessScriptApplication,
)

if TYPE_CHECKING:
    from .preferences import Mosplat_AP_Global
else:
    Mosplat_AP_Global: TypeAlias = Any

D = TypeVar("D", bound=DataclassInstance)


class MosplatPropertyGroupBase(
    Generic[D],
    PropertyGroup,
    MosplatBlPropertyAccessorMixin,
    MosplatDataclassInteropMixin[D],
):
    pass


class Mosplat_PG_PreprocessScriptApplication(
    MosplatPropertyGroupBase[PreprocessScriptApplication]
):
    __dataclass_type__ = PreprocessScriptApplication

    script_path: StringProperty(name="Script Path", subtype="FILE_PATH")
    application_time: FloatProperty(name="Application Time", default=-1.0)


class Mosplat_PG_ProcessedFrameRange(MosplatPropertyGroupBase[ProcessedFrameRange]):
    __dataclass_type__ = ProcessedFrameRange

    start_frame: IntProperty(name="Start Frame", default=0, min=0)
    end_frame: IntProperty(name="End Frame", default=0, min=0)
    applied_preprocess_scripts: CollectionProperty(
        name="Applied Preprocess Scripts", type=Mosplat_PG_PreprocessScriptApplication
    )

    @property
    def scripts_accessor(
        self,
    ) -> SupportsCollectionProperty[Mosplat_PG_PreprocessScriptApplication]:
        return self.applied_preprocess_scripts


class Mosplat_PG_MediaFileStatus(MosplatPropertyGroupBase[MediaFileStatus]):
    __dataclass_type__ = MediaFileStatus

    filepath: StringProperty(name="Filepath", subtype="FILE_PATH")
    frame_count: IntProperty(name="Frame Count", default=-1)
    width: IntProperty(name="Width", default=-1)
    height: IntProperty(name="Height", default=-1)
    is_valid: BoolProperty(name="Is Valid", default=False)
    message: StringProperty(name="Message")
    mod_time: FloatProperty(name="Modification Time", default=-1.0)
    file_size: IntProperty(name="File Size", default=-1)


class Mosplat_PG_MediaIODataset(MosplatPropertyGroupBase[MediaIODataset]):
    __dataclass_type__ = MediaIODataset

    base_directory: StringProperty(
        name="Base Directory",
        description="Filepath to directory containing media files being processed.",
        default=str(Path.home()),
        subtype="DIR_PATH",
    )

    do_all_details_match: BoolProperty(
        name="Do All Details Match",
        description="Tracks whether the found media in the current media directory all"
        "have matching frame count, width, and height.",
        default=False,
    )

    common_frame_count: IntProperty(
        name="Collective Media Frame Count",
        description="Common frame count for media within the selected media directory.",
        default=-1,
    )

    common_width: IntProperty(
        name="Collective Media Frame Count",
        description="Common width for media within the selected media directory.",
        default=-1,
    )

    common_height: IntProperty(
        name="Collective Media Frame Count",
        description="Common height for media within the selected media directory.",
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


def update_current_media_dir(self: Mosplat_PG_Global, context: Context):
    OperatorIDEnum.run(
        bpy.ops, OperatorIDEnum.VALIDATE_MEDIA_FILE_STATUSES, "INVOKE_DEFAULT"
    )

    self.logger().info(f"'{self.get_prop_name('current_media_dir')}' updated.")


class Mosplat_PG_Global(MosplatPropertyGroupBase[GlobalData]):
    __dataclass_type__ = GlobalData

    current_media_dir: StringProperty(
        name="Media Directory",
        description="Filepath to directory containing media files to be processed.",
        default=str(Path.home()),
        subtype="DIR_PATH",
        update=update_current_media_dir,
    )

    current_frame_range: IntVectorProperty(
        name="Frame Range",
        description="Start and end frame of data to be processed.",
        size=2,
        default=(0, 60),
        min=0,
    )

    current_media_io_dataset: PointerProperty(
        name="Media IO Dataset",
        description="Dataset for all media I/O operations",
        type=Mosplat_PG_MediaIODataset,
        options={"SKIP_SAVE"},
    )

    @property
    def dataset_accessor(self) -> Mosplat_PG_MediaIODataset:
        return self.current_media_io_dataset

    @property
    def current_media_dirpath(self) -> Path:
        return check_current_media_dirpath(self)

    def data_output_dirpath(self, prefs: Mosplat_AP_Global) -> Path:
        return check_data_output_dirpath(prefs, self)

    def data_json_filepath(self, prefs: Mosplat_AP_Global) -> Path:
        return check_data_json_filepath(prefs, self)

    def media_files(self, prefs: Mosplat_AP_Global) -> List[Path]:
        return check_media_files(prefs, self)
