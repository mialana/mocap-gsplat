import bpy
from typing import ClassVar
import string


class MosplatOperatorMixin:
    """A mixin that standardizes `bl_info` for Mosplat."""

    short_name: ClassVar[str]

    @classmethod
    def setup_bl_info(cls):
        if not cls.short_name:
            raise TypeError(f"{cls} does not declare required `short_name` variable.")

        cls.bl_idname = f"mosplat.{cls.short_name}"
        cls.bl_label = string.capwords(cls.bl_idname.replace(".", " "))


class Mosplat_OT_Base(MosplatOperatorMixin, bpy.types.Operator):
    pass
