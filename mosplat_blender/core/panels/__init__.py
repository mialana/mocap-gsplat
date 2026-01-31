from typing import List, Type

from .base_pt import MosplatPanelBase, MosplatUIListBase
from .main_pt import Mosplat_PT_Main
from .preprocess_pt import Mosplat_PT_Preprocess
from .log_entries_pt import Mosplat_PT_LogEntries, Mosplat_UL_log_entries

all_panels: List[Type[MosplatPanelBase]] = [
    Mosplat_PT_Main,
    Mosplat_PT_Preprocess,
    Mosplat_PT_LogEntries,
]

all_ui_lists: List[Type[MosplatUIListBase]] = [Mosplat_UL_log_entries]
