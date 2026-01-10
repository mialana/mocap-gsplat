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
    PointerProperty
    
)
import os
import logging


class MosplatProperties(PropertyGroup):

    logging_output: StringProperty(
        name="Logging Output",
        description="",
        default=bpy.utils.user_resource(
            "EXTENSIONS", path=os.path.join(".cache", "mosplat_blender", "log", "mosplat_%Y-%m-%d_%H-%M-%S.log")
        ),
    )