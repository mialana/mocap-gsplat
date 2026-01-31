from __future__ import annotations

from bpy.types import bpy_prop_array

from typing import TYPE_CHECKING, List, cast, Dict

from ...infrastructure.constants import MAX_LOG_ENTRIES_STORED
from ...infrastructure.protocols import SupportsCollectionProperty
from ...infrastructure.schemas import PanelIDEnum, UIListIDEnum, LogEntryLevelEnum

from .base_pt import MosplatPanelBase, MosplatUIListBase

if TYPE_CHECKING:
    from ..properties import Mosplat_PG_LogEntry
    from bpy.stub_internal.rna_enums import IconItems

LOG_LEVEL_ICON_MAP: Dict[LogEntryLevelEnum, IconItems] = {
    LogEntryLevelEnum.DEBUG: "NODE_SOCKET_MATRIX",
    LogEntryLevelEnum.INFO: "NODE_SOCKET_SHADER",
    LogEntryLevelEnum.WARNING: "NODE_SOCKET_RGBA",
    LogEntryLevelEnum.ERROR: "RECORD_ON",
    LogEntryLevelEnum.EXCEPTION: "RECORD_ON",
}


class Mosplat_UL_log_entries(MosplatUIListBase):
    bl_idname = UIListIDEnum.LOG_ENTRIES
    bl_description = "Organizes log entries for UI list."

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
        row.alert = (
            log.level_enum == LogEntryLevelEnum.ERROR
            or log.level_enum == LogEntryLevelEnum.EXCEPTION
        )
        split = row.split(factor=0.1, align=True)
        split.label(text=str(index))

        inner = split.split(factor=0.15, align=True)
        inner.label(
            text=log.level, icon=LOG_LEVEL_ICON_MAP[LogEntryLevelEnum[log.level]]
        )

        sub = inner.row()
        sub.label(text=log.message)

    def filter_items(self, context, data, property):
        collection: SupportsCollectionProperty[Mosplat_PG_LogEntry] = getattr(
            data, property
        )
        props = self.props(context)
        filter = props.log_level_filter_enum

        flt_flags: List[int] = []
        flt_neworder: List[int] = []

        for idx, log in enumerate(collection):
            if filter == LogEntryLevelEnum._ALL:
                flt_flags.append(self.bitflag_filter_item)
            elif log.level_enum == filter:
                flt_flags.append(self.bitflag_filter_item)
            else:
                flt_flags.append(self.bitflag_item_never_show)
            flt_neworder.append(idx)

        return cast(bpy_prop_array, flt_flags), cast(bpy_prop_array, flt_neworder)

    def draw_filter(self, context, layout) -> None:
        layout.row(align=True)

        props = self.props(context)
        layout.prop(props, props.get_prop_id("current_log_level_filter"))


class Mosplat_PT_LogEntries(MosplatPanelBase):
    bl_idname = PanelIDEnum.LOG_ENTRIES
    bl_description = "Panel to display log entries."
    bl_parent_id = PanelIDEnum.MAIN

    def draw_with_layout(self, pkg, layout):
        props = pkg.props

        template_list = layout.template_list(
            UIListIDEnum.LOG_ENTRIES,
            "",
            props,
            props.get_prop_id("current_log_entries"),
            props,
            props.get_prop_id("current_log_entry_index"),
            item_dyntip_propname="message",
            rows=MAX_LOG_ENTRIES_STORED // 2,
            sort_lock=True,
        )
