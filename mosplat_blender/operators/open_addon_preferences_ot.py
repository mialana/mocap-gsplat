from infrastructure.schemas import AddonMeta
from operators.base_ot import MosplatOperatorBase


class Mosplat_OT_open_addon_preferences(MosplatOperatorBase):
    def _contexted_execute(self, pkg):
        from bpy import ops

        context = pkg.context

        ops.screen.userpref_show()

        wm = self.wm(context)
        wm.addon_search = AddonMeta().human_readable_name
        wm.addon_filter = "All"

        if context.preferences:
            context.preferences.active_section = "ADDONS"

        return "FINISHED"
