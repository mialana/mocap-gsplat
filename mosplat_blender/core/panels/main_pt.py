from core.panels.base_pt import MosplatPanelBase
from infrastructure.schemas import OperatorIDEnum


class Mosplat_PT_Main(MosplatPanelBase):
    def draw_with_layout(self, pkg, layout):
        layout.operator(OperatorIDEnum.OPEN_ADDON_PREFERENCES, icon="SETTINGS")
