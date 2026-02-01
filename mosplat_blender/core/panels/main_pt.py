from ...infrastructure.schemas import OperatorIDEnum
from .base_pt import MosplatPanelBase


class Mosplat_PT_Main(MosplatPanelBase):
    def draw_with_layout(self, pkg, layout):
        layout.operator(OperatorIDEnum.OPEN_ADDON_PREFERENCES, icon="SETTINGS")
