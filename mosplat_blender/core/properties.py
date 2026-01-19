# pyright: reportInvalidTypeForm=false
from __future__ import annotations
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

from ..interfaces import MosplatLoggingInterface

logger = MosplatLoggingInterface.configure_logger_instance(__name__)


class Mosplat_PG_MediaItem(PropertyGroup):
    filepath: StringProperty(name="Filepath", subtype="FILE_PATH")
    frame_count: IntProperty(name="Frame Count", default=-1)


class Mosplat_PG_Global(PropertyGroup):
    current_media_dir: StringProperty(
        name="Media Directory",
        description="Filepath to directory containing media files to be processed.",
        default=str(Path.home()),
        subtype="DIR_PATH",
    )

    current_frame_range: IntVectorProperty(
        name="Frame Range",
        description="Start and end frame of data to be processed.",
        size=2,
        default=(0, 60),
        min=0,
    )

    computed_media_frame_count: IntProperty(
        name="Computed Media Frame Count",
        description="Shared frame count for media within the selected media directory.",
        default=-1,
    )

    do_media_durations_all_match: BoolProperty(
        name="Do Media Durations All Match",
        description="Tracks whether the found media in the current media directory all have matching durations.",
        default=False,
    )

    found_media_files: CollectionProperty(
        name="Found Media Files", type=Mosplat_PG_MediaItem
    )
