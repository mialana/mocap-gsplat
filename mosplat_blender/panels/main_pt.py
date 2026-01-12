import bpy
import logging

from ..infrastructure.bases import Mosplat_PT_Base


class Main_PT(Mosplat_PT_Base):
    short_name = "Main"
    parent_class = None

    @classmethod
    def poll(cls, context):
        return True

    def draw(self, context):
        if not (layout := self.layout):
            return


class Child_PT(Mosplat_PT_Base):
    short_name = "Child"
    parent_class = Main_PT

    @classmethod
    def poll(cls, context):
        return True

    def draw(self, context):
        if not (layout := self.layout):
            return
