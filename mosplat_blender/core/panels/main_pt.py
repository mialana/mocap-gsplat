from ...infrastructure.identifiers import OperatorIDEnum
from .base_pt import MosplatPanelBase


class Mosplat_PT_main(MosplatPanelBase):
    def draw_with_layout(self, pkg, layout):
        layout.operator(OperatorIDEnum.OPEN_ADDON_PREFERENCES, icon="SETTINGS")

        prefs = pkg.prefs
        row = layout.row()
        row.alignment = "CENTER"
        row.prop(prefs, prefs._meta.force_all_operations.id)
