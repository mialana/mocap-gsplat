from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, cast

from bpy.types import bpy_prop_array

from ...infrastructure.constants import DEFAULT_LOG_ENTRY_ROWS
from ...infrastructure.protocols import SupportsCollectionProperty
from ...infrastructure.schemas import LogEntryLevelEnum, UIListIDEnum
from ..properties import Mosplat_PG_LogEntry, Mosplat_PG_LogEntryHub
from .base_pt import MosplatPanelBase, MosplatUIListBase

if TYPE_CHECKING:
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
            log.level_as_enum == LogEntryLevelEnum.ERROR
            or log.level_as_enum == LogEntryLevelEnum.EXCEPTION
        )
        split = row.split(factor=0.15, align=True)
        split.label(
            text=log.level, icon=LOG_LEVEL_ICON_MAP[LogEntryLevelEnum[log.level]]
        )

        inner = split.split(factor=0.9, align=True)

        inner.label(text=log.message)
        sub = inner.row(align=True)
        sub.alignment = "RIGHT"
        sub.label(text=str(index))

    def filter_items(self, context, data, property):
        collection: SupportsCollectionProperty[Mosplat_PG_LogEntry] = getattr(
            data, property
        )
        log_hub = self.props(context).log_hub_accessor
        filter = log_hub.logs_level_filter

        flt_flags: List[int] = []
        flt_neworder: List[int] = []

        for idx, log in enumerate(collection):
            if filter == LogEntryLevelEnum.ALL:
                flt_flags.append(self.bitflag_filter_item)
            elif log.level_as_enum == filter:
                flt_flags.append(self.bitflag_filter_item)
            else:
                flt_flags.append(self.bitflag_item_never_show)
            flt_neworder.append(idx)

        return cast(bpy_prop_array, flt_flags), cast(bpy_prop_array, flt_neworder)

    def draw_filter(self, context, layout) -> None:
        layout.row(align=True)

        log_hub = self.props(context).log_hub_accessor
        layout.prop(log_hub, Mosplat_PG_LogEntryHub.level_filter_prop_id())


class Mosplat_PT_LogEntries(MosplatPanelBase):
    def draw_with_layout(self, pkg, layout):
        log_hub = pkg.props.log_hub_accessor

        layout.template_list(
            UIListIDEnum.LOG_ENTRIES,
            "",
            log_hub,
            Mosplat_PG_LogEntryHub.data_prop_id(),
            log_hub,
            Mosplat_PG_LogEntryHub.active_index_prop_id(),
            item_dyntip_propname=Mosplat_PG_LogEntry.dyntip_prop_id(),
            rows=DEFAULT_LOG_ENTRY_ROWS,
            sort_lock=True,
        )
