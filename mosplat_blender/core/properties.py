# pyright: reportInvalidTypeForm=false
from __future__ import annotations

import bpy
from bpy.types import PropertyGroup, Context
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
    CollectionProperty,
    PointerProperty,
    IntVectorProperty,
)

from pathlib import Path

from ..infrastructure.mixins import (
    MosplatBlPropertyAccessorMixin,
    MosplatDataclassInteropMixin,
)
from ..infrastructure.constants import OperatorIDEnum
from ..infrastructure.schemas import (
    GlobalData,
    MediaIOMetadata,
    MediaProcessStatus,
    ProcessedFrameRange,
    PreprocessScriptApplication,
)


def update_current_media_dir(props: Mosplat_PG_Global, _: Context):
    OperatorIDEnum.run(bpy.ops, OperatorIDEnum.CHECK_MEDIA_FRAME_COUNTS)


class MosplatPropertyGroupBase(
    PropertyGroup, MosplatBlPropertyAccessorMixin, MosplatDataclassInteropMixin
):
    pass


class Mosplat_PG_PreprocessScriptApplication(MosplatPropertyGroupBase):
    __dataclass_type__ = PreprocessScriptApplication

    script_path: StringProperty(name="Script Path", subtype="FILE_PATH")
    application_time: FloatProperty(name="Application Time", default=-1.0)


class Mosplat_PG_ProcessedFrameRange(MosplatPropertyGroupBase):
    __dataclass_type__ = ProcessedFrameRange

    start_frame: IntProperty(name="Start Frame", default=0, min=0)
    end_frame: IntProperty(name="End Frame", default=0, min=0)
    applied_preprocess_scripts: CollectionProperty(
        name="Applied Preprocess Scripts", type=Mosplat_PG_PreprocessScriptApplication
    )


class Mosplat_PG_MediaProcessStatus(MosplatPropertyGroupBase):
    __dataclass_type__ = MediaProcessStatus

    filepath: StringProperty(name="Filepath", subtype="FILE_PATH")
    frame_count: IntProperty(name="Frame Count", default=-1)
    is_valid: BoolProperty(name="Is Valid", default=False)
    message: StringProperty(name="Message")
    mod_time: FloatProperty(name="Modification Time", default=-1.0)
    file_size: IntProperty(name="File Size", default=-1)


class Mosplat_PG_MediaIOMetadata(MosplatPropertyGroupBase):
    __dataclass_type__ = MediaIOMetadata

    base_directory: StringProperty(
        name="Base Directory",
        description="Filepath to directory containing media files being processed.",
        default=str(Path.home()),
        subtype="DIR_PATH",
        update=update_current_media_dir,
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


class Mosplat_PG_Global(MosplatPropertyGroupBase):
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

    was_restored_from_json: BoolProperty(
        name="Was Restored From JSON",
        description="Checked during all `poll()` methods of operators.\n"
        "When false at the beginning of the session, property data will be restored from JSON.",
        default=False,
        options={"SKIP_SAVE"},
    )
