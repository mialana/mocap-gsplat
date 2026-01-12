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
from .infrastructure.logs import MosplatLoggingManager

logger = MosplatLoggingManager.configure_logger_instance(__name__)


class Mosplat_Properties(PropertyGroup):
    pass
