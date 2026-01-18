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


def check_media_durations(props: Mosplat_PG_Global, _: Context):
    pass


class Mosplat_PG_Global(PropertyGroup):
    current_media_dir: StringProperty(
        name="Media Directory",
        description="Filepath to directory containing media files to be processed.",
        default=str(Path.home()),
        subtype="DIR_PATH",
        update=check_media_durations,
    )

    current_frame_range: IntVectorProperty(
        name="Frame Range",
        description="Start and end frame of data to be processed.",
        size=2,
        default=(0, 60),
        min=0,
    )

    do_media_durations_all_match: BoolProperty(
        name="Do Media Durations All Match",
        description="Tracks whether the found media in the current media directory all have matching durations.",
        default=False,
    )
