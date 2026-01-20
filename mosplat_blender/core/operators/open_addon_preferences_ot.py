import bpy
from bpy.props import StringProperty

from ...infrastructure.constants import OperatorIDEnum, ADDON_HUMAN_READABLE

from .base_ot import MosplatOperatorBase, OperatorReturnItemsSet, OperatorPollReqs


class Mosplat_OT_open_addon_preferences(MosplatOperatorBase):
    bl_idname = OperatorIDEnum.OPEN_ADDON_PREFERENCES
    bl_description = (
        f"Quick navigation to {ADDON_HUMAN_READABLE} saved addon preferences."
    )

    __poll_reqs__ = {OperatorPollReqs.WINDOW_MANAGER}

    def execute(self, context) -> OperatorReturnItemsSet:
        bpy.ops.screen.userpref_show()

        wm = self.wm(context)
        wm.addon_search = ADDON_HUMAN_READABLE
        wm.addon_filter = "All"

        if context.preferences:
            context.preferences.active_section = "ADDONS"

        return {"FINISHED"}
