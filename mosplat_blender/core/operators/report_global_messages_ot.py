from bpy.props import StringProperty, EnumProperty

from string import capwords
from typing import TypeAlias, Tuple, Final, List, get_args
from ...infrastructure.schemas import (
    OperatorIDEnum,
    WmReportItems,
    WmReportItemsEnum,
)

from .base_ot import MosplatOperatorBase

WmReportEnumPropertyItem: TypeAlias = Tuple[WmReportItems, str, str]
WmReportEnumPropertyList: Final[List[WmReportEnumPropertyItem]] = [
    (item, capwords(item.lower().replace("_", " ")), "")
    for item in get_args(WmReportItems)
]


class Mosplat_OT_report_global_messages(MosplatOperatorBase):
    bl_idname = OperatorIDEnum.REPORT_GLOBAL_MESSAGES
    bl_description = "Reports logging messages to Blender's window manager."

    message: StringProperty()  # type: ignore[reportInvalidTypeForm]
    level: EnumProperty(
        items=WmReportEnumPropertyList,
        default=WmReportItemsEnum.INFO.value,
    )  # type: ignore[reportInvalidTypeForm]

    def _contexted_invoke(self, pkg, event):
        wm = self.wm(pkg.context)

        return wm.invoke_props_dialog(self, width=200)

    def _contexted_execute(self, pkg):
        self._level: WmReportItems = self.level
        self.report({self._level}, self.message)

        return "FINISHED"
