from __future__ import annotations

from bpy.props import StringProperty
from typing import TYPE_CHECKING
from ...infrastructure.schemas import OperatorIDEnum
from .base_ot import MosplatOperatorBase

if TYPE_CHECKING:
    from .base_ot import OpResultSet


class Mosplat_OT_report_global_messages(MosplatOperatorBase):
    bl_idname = OperatorIDEnum.REPORT_GLOBAL_MESSAGES
    bl_description = "Reports logging messages to Blender's window manager."
    bl_options = {"REGISTER"}

    message: StringProperty()  # type: ignore[reportInvalidTypeForm]

    def invoke(self, context, event) -> OpResultSet:
        """override original methods as this op does not need the same monitoring"""
        return self.execute(context)

    def execute(self, context) -> OpResultSet:
        """override original methods as this op does not need the same monitoring"""

        self.report({"INFO"}, self.message)

        return {"FINISHED"}
