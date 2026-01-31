from __future__ import annotations

from bpy.props import StringProperty, EnumProperty

from string import capwords
from typing import TypeAlias, Tuple, Final, List, get_args, TYPE_CHECKING
from ...infrastructure.schemas import OperatorIDEnum, WmReportItems, WmReportItemsEnum

from .base_ot import MosplatOperatorBase

if TYPE_CHECKING:
    from .base_ot import OpResultSet


WmReportEnumPropertyItem: TypeAlias = Tuple[WmReportItems, str, str]
WmReportEnumPropertyList: Final[List[WmReportEnumPropertyItem]] = [
    (item, capwords(item.lower().replace("_", " ")), "")
    for item in get_args(WmReportItems)
]


class Mosplat_OT_report_global_messages(MosplatOperatorBase):
    bl_idname = OperatorIDEnum.REPORT_GLOBAL_MESSAGES
    bl_description = "Reports logging messages to Blender's window manager."
    bl_options = {"REGISTER", "MACRO"}

    message: StringProperty()  # type: ignore[reportInvalidTypeForm]
    level: EnumProperty(
        items=WmReportEnumPropertyList,
        default=WmReportItemsEnum.INFO.value,
    )  # type: ignore[reportInvalidTypeForm]

    def invoke(self, context, event) -> OpResultSet:
        """override native methods as this op does not need the same monitoring"""
        wm = self.wm(context)

        return wm.invoke_props_dialog(self, width=200)

    def execute(self, context) -> OpResultSet:
        """override native methods as this op does not need the same monitoring"""
        self._level: WmReportItems = self.level
        self.report({self._level}, self.message)

        return {"FINISHED"}
