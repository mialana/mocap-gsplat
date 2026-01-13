import bpy

from .base import MosplatPanelBase


class Mosplat_PT_Main(MosplatPanelBase):
    short_name = "Main"
    parent_class = None

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
