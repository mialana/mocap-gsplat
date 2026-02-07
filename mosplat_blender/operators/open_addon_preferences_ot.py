from typing import NamedTuple, Tuple

from infrastructure.schemas import AddonMeta
from operators.base_ot import MosplatOperatorBase


class ProcessKwargs(NamedTuple):
    pass


class Mosplat_OT_open_addon_preferences(
    MosplatOperatorBase[Tuple[str, int], ProcessKwargs]
):
    def _contexted_execute(self, pkg):
        from bpy import ops

        context = pkg.context

        ops.screen.userpref_show()

        wm = self.wm(context)
        wm.addon_search = AddonMeta().human_readable_name
        wm.addon_filter = "All"

        if context.preferences:
            context.preferences.active_section = "ADDONS"

        self.launch_subprocess(
            pkg.context,
            pwargs=ProcessKwargs(),
        )

        return "RUNNING_MODAL"

    def _queue_callback(self, pkg, event, next):
        status, payload = next
        self.logger.warning(status)
        self.logger.warning(payload)

        if status == "done":
            self.logger.warning("finished")
            return "FINISHED"
        return "RUNNING_MODAL"

    @staticmethod
    def _operator_subprocess(queue, cancel_event, *, pwargs):
        import torchcodec

        queue.put(("hello from process", 5))

        import time

        time.sleep(5)

        queue.put(("done", 10))
        return


def process_entrypoint(*args, **kwargs):
    Mosplat_OT_open_addon_preferences._operator_subprocess(*args, **kwargs)
