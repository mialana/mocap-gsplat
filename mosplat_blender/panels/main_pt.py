import bpy
import logging

from ..types import Mosplat_PT_Base


class Main_PT(Mosplat_PT_Base):
    short_name = "Main"
    parent_class = None

    @classmethod
    def poll(cls, context):
        cls.logger.debug("Polled")
        return True

    def draw(self, context):
        self.logger.debug(f"Draw called.")


class Child_PT(Mosplat_PT_Base):
    short_name = "Child"
    parent_class = Main_PT

    @classmethod
    def poll(cls, context):
        cls.logger.debug("Polled")
        return True

    def draw(self, context):
        if not (layout := self.layout):
            return

        self.logger.debug("Draw called.")
