import bpy

from ...infrastructure.mixins import MosplatBlMetaPanelMixin


class MosplatPanelBase(MosplatBlMetaPanelMixin, bpy.types.Panel):
    prefix_suffix = "PT"

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
