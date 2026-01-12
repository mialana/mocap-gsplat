import bpy

from .mixins import MosplatBlMetaMixin, MosplatBlMetaPanelMixin


class Mosplat_PT_Base(MosplatBlMetaPanelMixin, bpy.types.Panel):
    prefix_suffix = "PT"

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"


class Mosplat_OT_Base(MosplatBlMetaMixin, bpy.types.Operator):
    prefix_suffix = "OT"
