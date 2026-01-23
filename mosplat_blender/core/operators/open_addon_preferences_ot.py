import bpy

from ...infrastructure.constants import ADDON_HUMAN_READABLE
from ...infrastructure.schemas import OperatorIDEnum

from .base_ot import (
    MosplatOperatorBase,
    OperatorReturnItemsSet,
)


class Mosplat_OT_open_addon_preferences(MosplatOperatorBase):
    bl_idname = OperatorIDEnum.OPEN_ADDON_PREFERENCES
    bl_description = (
        f"Quick navigation to {ADDON_HUMAN_READABLE} saved addon preferences."
    )

    def contexted_execute(self, context) -> OperatorReturnItemsSet:
        bpy.ops.screen.userpref_show()

        wm = self.wm
        wm.addon_search = ADDON_HUMAN_READABLE
        wm.addon_filter = "All"

        if context.preferences:
            context.preferences.active_section = "ADDONS"

        return {"FINISHED"}
