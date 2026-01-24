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
    check_metadata_json_filepath,
    check_current_media_dirpath,
)

from ..infrastructure.mixins import (
    MosplatBlPropertyAccessorMixin,
    MosplatDataclassInteropMixin,
)
from ..infrastructure.constants import DataclassInstance
from ..infrastructure.schemas import (
    OperatorIDEnum,
    GlobalData,
    MediaIOMetadata,
    MediaProcessStatus,
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


class Mosplat_PG_MediaProcessStatus(MosplatPropertyGroupBase[MediaProcessStatus]):
    __dataclass_type__ = MediaProcessStatus

    filepath: StringProperty(name="Filepath", subtype="FILE_PATH")
    frame_count: IntProperty(name="Frame Count", default=-1)
    is_valid: BoolProperty(name="Is Valid", default=False)
    message: StringProperty(name="Message")
    mod_time: FloatProperty(name="Modification Time", default=-1.0)
    file_size: IntProperty(name="File Size", default=-1)


class Mosplat_PG_MediaIOMetadata(MosplatPropertyGroupBase[MediaIOMetadata]):
    __dataclass_type__ = MediaIOMetadata

    base_directory: StringProperty(
        name="Base Directory",
        description="Filepath to directory containing media files being processed.",
        default=str(Path.home()),
        subtype="DIR_PATH",
    )

    do_media_durations_all_match: BoolProperty(
        name="Do Media Durations All Match",
        description="Tracks whether the found media in the current media directory all have matching durations.",
        default=False,
    )

    collective_media_frame_count: IntProperty(
        name="Collective Media Frame Count",
        description="Shared frame count for media within the selected media directory.",
        default=-1,
    )

    media_process_statuses: CollectionProperty(
        name="Found Media Files", type=Mosplat_PG_MediaProcessStatus
    )

    processed_frame_ranges: CollectionProperty(
        name="Processed Frame Ranges", type=Mosplat_PG_ProcessedFrameRange
    )

    def to_JSON(self, json_filepath):
        self.to_dataclass().to_JSON(json_filepath)


def update_current_media_dir(self: Mosplat_PG_Global, context: Context):
    OperatorIDEnum.run(
        bpy.ops, OperatorIDEnum.CHECK_MEDIA_FRAME_COUNTS, "INVOKE_DEFAULT"
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

    current_media_io_metadata: PointerProperty(
        name="Media IO Metadata",
        description="Metadata for all media I/O operations",
        type=Mosplat_PG_MediaIOMetadata,
        options={"SKIP_SAVE"},
    )

    @property
    def metadata_ptr(self) -> Mosplat_PG_MediaIOMetadata:
        return self.current_media_io_metadata

    @property
    def current_media_dirpath(self) -> Path:
        return check_current_media_dirpath(self)

    def data_output_dirpath(self, prefs: Mosplat_AP_Global) -> Path:
        return check_data_output_dirpath(prefs, self)

    def metadata_json_filepath(self, prefs: Mosplat_AP_Global) -> Path:
        return check_metadata_json_filepath(prefs, self)

    def media_files(self, prefs: Mosplat_AP_Global) -> List[Path]:
        return check_media_files(prefs, self)
