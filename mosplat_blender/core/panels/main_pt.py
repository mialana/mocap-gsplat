import bpy

from .base import MosplatPanelBase


class Mosplat_PT_Main(MosplatPanelBase):
    short_name = "Main"
    parent_class = None

    bl_options = {"HEADER_LAYOUT_EXPAND"}

    @classmethod
    def poll(cls, context):
        return True

    def draw(self, context):
        if not (layout := self.layout):
            return


class Mosplat_PT_Child(MosplatPanelBase):
    short_name = "Child"
    parent_class = Mosplat_PT_Main

    @classmethod
    def poll(cls, context):
        return True

    def draw(self, context):
        if not (layout := self.layout):
            return

        row = layout.row()
        row.operator("mosplat.install_model")
