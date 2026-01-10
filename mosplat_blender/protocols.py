import bpy
import inspect
from typing import Protocol, Any, runtime_checkable

from .properties import MosplatProperties

class SceneProtocol(Protocol):
    mosplat_properties: MosplatProperties

    @classmethod
    def assign_property(cls, scene: bpy.types.Scene, name: str, value: Any):
        all_prop_types = inspect.getmembers(bpy.props)

        if not any(isinstance(value, p) for (name, p) in all_prop_types):
            raise TypeError("Attributes set on Scene should be a valid Blender property type.")
        
        setattr(scene, name, value)
