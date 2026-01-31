from typing import TYPE_CHECKING

from ...infrastructure.schemas import PanelIDEnum, UIListIDEnum

from .base_pt import MosplatPanelBase, MosplatUIListBase

if TYPE_CHECKING:
    from ..properties import Mosplat_PG_LogEntry


class Mosplat_UL_log_entries(MosplatUIListBase):
    bl_idname = UIListIDEnum.LOG_ENTRIES

    def draw_item(
        self,
        context,
        layout,
        data,
        item,
        icon,
        active_data,
        active_property,
        index,
        flt_flag,
    ):
        if not item:
            return
        log: Mosplat_PG_LogEntry = item

        row = layout.row(align=True)
        row.label(text=log.level)
        row.prop(log, "message", text="", emboss=False)


class Mosplat_PT_LogEntries(MosplatPanelBase):
    bl_idname = PanelIDEnum.LOG_ENTRIES
    bl_description = "Panel to display log entries."
    bl_parent_id = PanelIDEnum.MAIN
    bl_order = 10

    def draw_with_layout(self, pkg, layout):
        props = pkg.props

        template_list = layout.template_list(
            UIListIDEnum.LOG_ENTRIES,  # listtype_name
            "",  # list_id
            props,  # dataptr
            "current_log_entries",  # propname
            props,  # active_dataptr
            "current_log_entry_index",  # active_propname
            rows=6,
        )
