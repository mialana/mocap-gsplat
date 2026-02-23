from ..infrastructure.schemas import AddonMeta
from .base_ot import MosplatOperatorBase


class Mosplat_OT_fit_timeline(MosplatOperatorBase):
    def _contexted_execute(self, pkg):
        import bpy
        from bpy.types import Area, Region, Window

        context = pkg.context

        dopesheet_found = False
        wm = self.wm(context)
        window: Window
        for window in wm.windows:
            screen = window.screen
            if not screen:
                continue
            area: Area
            for area in screen.areas:
                if area.type == "DOPESHEET_EDITOR":
                    region: Region
                    for region in area.regions:
                        if region.type == "WINDOW":
                            _ovr = {"window": window, "area": area, "region": region}
                            with context.temp_override(**_ovr):
                                bpy.ops.action.view_all()
                            dopesheet_found = True

        if not dopesheet_found:
            self.logger.warning(
                "Dopesheet not found in UI, could not fit timeline to frame range."
            )

        return "FINISHED"
