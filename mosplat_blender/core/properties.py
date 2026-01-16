# pyright: reportInvalidTypeForm=false

import bpy
from bpy.types import PropertyGroup
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
    CollectionProperty,
    PointerProperty,
)

from pathlib import Path
from ..interfaces import MosplatLoggingInterface

logger = MosplatLoggingInterface.configure_logger_instance(__name__)


class Mosplat_PG_Global(PropertyGroup):
    current_media_dir: StringProperty(
        name="Current Media Directory",
        description="Filepath to directory containing media files to be processed.",
        default=str(Path.home()),
        subtype="DIR_PATH",
    )
