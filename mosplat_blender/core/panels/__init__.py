from typing import List, Type

from .base import MosplatPanelBase
from . import main_pt

all_panels: List[Type[MosplatPanelBase]] = [
    main_pt.Mosplat_PT_Main,
    main_pt.Mosplat_PT_Child,
]
