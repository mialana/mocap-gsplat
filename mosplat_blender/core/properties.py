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
