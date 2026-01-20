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
from typing import cast

from ..infrastructure.mixins import MosplatBlPropertyAccessorMixin
from ..infrastructure.constants import OperatorIDEnum
from ..infrastructure.schemas import (
    MediaIOMetadata,
    MediaProcessStatus,
    ProcessedFrameRange,
    PreprocessScriptApplication,
)


def update_current_media_dir(props: Mosplat_PG_Global, _: Context):
    OperatorIDEnum.run(bpy.ops, OperatorIDEnum.CHECK_MEDIA_FRAME_COUNTS)


class Mosplat_PG_PreprocessScriptApplication(PropertyGroup):
    script_path: StringProperty(name="Script Path", subtype="FILE_PATH")
    application_time: FloatProperty(name="Application Time", default=-1.0)

    def to_dataclass(self) -> PreprocessScriptApplication:
        return PreprocessScriptApplication(
            script_path=self.script_path,
            application_time=self.application_time,
        )

    def from_dataclass(self) -> None:
        pass


class Mosplat_PG_MediaProcessStatus(PropertyGroup):
    filepath: StringProperty(name="Filepath", subtype="FILE_PATH")
    frame_count: IntProperty(name="Frame Count", default=-1)
    is_valid: BoolProperty(name="Is Valid", default=False)
    message: StringProperty(name="Message")
    mod_time: FloatProperty(name="Modification Time", default=-1.0)
    file_size: IntProperty(name="File Size", default=-1)

    def to_dataclass(self) -> MediaProcessStatus:
        return MediaProcessStatus(
            filepath=self.filepath,
            frame_count=self.frame_count,
            is_valid=self.is_valid,
            message=self.message,
            mod_time=self.mod_time,
            file_size=self.file_size,
        )


class Mosplat_PG_ProcessedFrameRange(PropertyGroup):
    start_frame: IntProperty(name="Start Frame", default=0, min=0)
    end_frame: IntProperty(name="End Frame", default=0, min=0)
    applied_preprocess_scripts: CollectionProperty(
        name="Applied Preprocess Scripts", type=Mosplat_PG_PreprocessScriptApplication
    )

    def to_dataclass(self) -> ProcessedFrameRange:
        return ProcessedFrameRange(
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            applied_preprocess_scripts=[
                cast(Mosplat_PG_PreprocessScriptApplication, a).to_dataclass()
                for a in self.applied_preprocess_scripts
            ],
        )


class Mosplat_PG_MediaIOMetadata(PropertyGroup):
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

    def to_dataclass(self) -> MediaIOMetadata:
        return MediaIOMetadata(
            base_directory=self.base_directory,
            do_media_durations_all_match=self.do_media_durations_all_match,
            collective_media_frame_count=self.collective_media_frame_count,
            media_process_statuses=[
                cast(Mosplat_PG_MediaProcessStatus, m).to_dataclass()
                for m in self.media_process_statuses
            ],
            processed_frame_ranges=[
                cast(Mosplat_PG_ProcessedFrameRange, p).to_dataclass()
                for p in self.processed_frame_ranges
            ],
        )


class Mosplat_PG_Global(PropertyGroup, MosplatBlPropertyAccessorMixin):
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

    collective_media_frame_count: IntProperty(
        name="Collective Media Frame Count",
        description="Shared frame count for media within the selected media directory.",
        default=-1,
    )

    do_media_durations_all_match: BoolProperty(
        name="Do Media Durations All Match",
        description="Tracks whether the found media in the current media directory all have matching durations.",
        default=False,
    )

    media_process_statuses: CollectionProperty(
        name="Found Media Files", type=Mosplat_PG_MediaProcessStatus
    )

    processed_frame_ranges: CollectionProperty(
        name="Processed Frame Ranges", type=Mosplat_PG_ProcessedFrameRange
    )
