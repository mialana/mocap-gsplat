from typing import List, Type

from ..infrastructure.bases import Mosplat_PT_Base
from . import main_pt

all_panels: List[Type[Mosplat_PT_Base]] = [main_pt.Main_PT, main_pt.Child_PT]
