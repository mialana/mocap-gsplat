from ...infrastructure.identifiers import OperatorIDEnum
from .base_pt import MosplatPanelBase


class Mosplat_PT_main(MosplatPanelBase):
    def draw_with_layout(self, pkg, layout):
        layout.operator(OperatorIDEnum.OPEN_ADDON_PREFERENCES, icon="SETTINGS")
