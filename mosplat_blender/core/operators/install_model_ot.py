from bpy.props import StringProperty

from pathlib import Path
import threading
from queue import Queue

from ...interfaces.vggt_interface import MosplatVGGTInterface

from ...infrastructure.constants import OperatorIDEnum

from .base import MosplatOperatorBase, OperatorReturnItemsSet


class Mosplat_OT_initialize_model(MosplatOperatorBase):
    bl_description = "Install VGGT model weights from Hugging Face."

    bl_idname = OperatorIDEnum.INITIALIZE_MODEL

    vggt_hf_id: StringProperty()  # pyright: ignore[reportInvalidTypeForm]
    vggt_outdir = StringProperty(
        subtype="DIR_PATH"
    )  # pyright: ignore[reportInvalidTypeForm]

    def modal_with_window_manager(self, context, event, wm) -> OperatorReturnItemsSet:
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        self.logger().debug("Polled via event timer!")

        if hasattr(self, "_queue") and not self._queue.empty():
            _, payload = self._queue.get_nowait()
            wm.event_timer_remove(self._timer)
            if payload:
                self.logger().info("Successfully initialized VGGT model!")
                return {"FINISHED"}
            else:
                self.logger().error("VGGT model could not be initialized")
                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke_with_window_manager(self, context, event, wm) -> OperatorReturnItemsSet:
        if not (prefs := self.prefs(context)):
            return {"CANCELLED"}

        self.vggt_hf_id = prefs.vggt_hf_id
        self.vggt_outdir = prefs.vggt_model_dir

        self._timer = wm.event_timer_add(time_step=0.1, window=context.window)

        return self.execute(context)

    def execute_with_window_manager(self, context, wm) -> OperatorReturnItemsSet:
        if not self.vggt_hf_id or not self.vggt_outdir:
            self.logger().error(
                "Call `invoke` to set up operator attributes before `execute`."
            )
            return {"CANCELLED"}

        vggt_outdir_path: Path = Path(self.vggt_outdir)  # convert to path here

        self._queue = Queue()
        self._thread = threading.Thread(
            target=self._install_model_thread,
            args=(self.vggt_hf_id, vggt_outdir_path),
            daemon=True,
        )
        self._thread.start()

        wm.modal_handler_add(self)  # start timer polling here

        return {"RUNNING_MODAL"}

    def _install_model_thread(self, hf_id: str, outdir: Path):
        if not hasattr(self, "_queue"):
            return

        # put true or false initialize result in queue
        MosplatVGGTInterface.initialize_model(hf_id, outdir)

        """
        use initialization status rather than return result as `initialize_model`
        will return `False` if initialization status already occurred"""
        self._queue.put(("ok", MosplatVGGTInterface._initialized))

        self.logger().debug("Install model thread completed!")
