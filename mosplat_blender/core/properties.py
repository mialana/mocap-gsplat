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
import os
from pathlib import Path
from ..interfaces import MosplatLoggingInterface

logger = MosplatLoggingInterface.configure_logger_instance(__name__)


class Mosplat_PG_Global(PropertyGroup):
    preset_enum: bpy.props.EnumProperty(
        name="",
        description="Select an option",
        items=[
            ("1", "AmbientCG", "https://ambientcg.com/"),
            ("2", "Texturify", "https://texturify.com/"),
            ("3", "BelderKit", "https://www.blenderkit.com/"),
        ],
    )

    current_image_dir: StringProperty(
        description="Filepath to directory of images that is currently being processsed",
        default=str(Path.home()),
        subtype="DIR_PATH",
    )
