from typing import List, Type, Dict, Tuple, Callable

from .base_pt import MosplatPanelBase, MosplatUIListBase, MosplatPanelMetadata
from .main_pt import Mosplat_PT_Main
from .preprocess_pt import Mosplat_PT_Preprocess
from .log_entries_pt import Mosplat_PT_LogEntries, Mosplat_UL_log_entries

from ...infrastructure.schemas import PanelIDEnum
from ...infrastructure.constants import ADDON_SHORTNAME

all_panels: List[Type[MosplatPanelBase]] = [
    Mosplat_PT_Main,
    Mosplat_PT_Preprocess,
    Mosplat_PT_LogEntries,
]

all_ui_lists: List[Type[MosplatUIListBase]] = [Mosplat_UL_log_entries]

panel_registry: Dict[
    Type[MosplatPanelBase],
    MosplatPanelMetadata,
] = {
    Mosplat_PT_Main: MosplatPanelMetadata(
        bl_idname=PanelIDEnum.MAIN,
        bl_description=f"Main panel holding all '{ADDON_SHORTNAME}' panels",
    ),
    Mosplat_PT_Preprocess: MosplatPanelMetadata(
        bl_idname=PanelIDEnum.PREPROCESS,
        bl_description=f"Holds operations for preprocessing '{ADDON_SHORTNAME}' data",
    ),
    Mosplat_PT_LogEntries: MosplatPanelMetadata(
        bl_idname=PanelIDEnum.LOG_ENTRIES,
        bl_description="Panel to display log entries.",
    ),
}

panel_factory: List[Tuple[Type[MosplatPanelBase], Callable[[], None]]] = [
    (cls, cls.preregistration_fn_factory(metadata))
    for cls, metadata in panel_registry.items()
]
