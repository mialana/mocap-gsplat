import bpy

from ...infrastructure.mixins import MosplatPanelMixin


class MosplatPanelBase(MosplatPanelMixin, bpy.types.Panel):
    pass
