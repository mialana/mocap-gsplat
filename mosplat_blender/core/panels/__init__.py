from typing import Callable, Dict, List, Tuple, Type

from core.panels.base_pt import (
    MosplatPanelBase,
    MosplatPanelMetadata,
    MosplatUIListBase,
    MosplatUIListMetadata,
)
from core.panels.log_entries_pt import Mosplat_PT_LogEntries, Mosplat_UL_log_entries
from core.panels.main_pt import Mosplat_PT_Main
from core.panels.preprocess_pt import Mosplat_PT_Preprocess
from infrastructure.identifiers import PanelIDEnum, UIListIDEnum
from infrastructure.mixins import PreregristrationFn
from infrastructure.schemas import AddonMeta

_ADDON_SHORTNAME = AddonMeta().shortname
panel_registry: Dict[
    Type[MosplatPanelBase],
    MosplatPanelMetadata,
] = {
    Mosplat_PT_Main: MosplatPanelMetadata(
        bl_idname=PanelIDEnum.MAIN,
        bl_description=f"Main panel holding all '{_ADDON_SHORTNAME}' panels",
        bl_options={"HIDE_HEADER"},
    ),
    Mosplat_PT_Preprocess: MosplatPanelMetadata(
        bl_idname=PanelIDEnum.PREPROCESS,
        bl_description=f"Holds operations for preprocessing '{_ADDON_SHORTNAME}' data",
        bl_parent_id=PanelIDEnum.MAIN,
    ),
    Mosplat_PT_LogEntries: MosplatPanelMetadata(
        bl_idname=PanelIDEnum.LOG_ENTRIES,
        bl_description="Panel to display log entries.",
        bl_parent_id=PanelIDEnum.MAIN,
    ),
}

panel_factory: List[Tuple[Type[MosplatPanelBase], Callable[[], None]]] = [
    (cls, cls.preregistration_fn_factory(metadata=mta))
    for cls, mta in panel_registry.items()
]


ui_list_registry: Dict[Type[MosplatUIListBase], MosplatUIListMetadata] = {
    Mosplat_UL_log_entries: MosplatUIListMetadata(
        bl_idname=UIListIDEnum.LOG_ENTRIES,
        bl_description="Organizes log entries for UI list.",
    )
}

ui_list_factory: List[Tuple[Type[MosplatUIListBase], PreregristrationFn]] = [
    (cls, cls.preregistration_fn_factory(metadata=mta))
    for cls, mta in ui_list_registry.items()
]
