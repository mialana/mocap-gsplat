import bpy

from core.operators.base_ot import MosplatOperatorBase
from infrastructure.constants import ADDON_HUMAN_READABLE


class Mosplat_OT_open_addon_preferences(MosplatOperatorBase):
    def _contexted_execute(self, pkg):
        context = pkg.context

        bpy.ops.screen.userpref_show()

        wm = self.wm(context)
        wm.addon_search = ADDON_HUMAN_READABLE
        wm.addon_filter = "All"

        if context.preferences:
            context.preferences.active_section = "ADDONS"

        return "FINISHED"
