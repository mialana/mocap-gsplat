from bpy.types import Operator, Context
from bpy.props import StringProperty

from pathlib import Path
import threading
from queue import Queue

from ...interfaces.vggt_interface import MosplatVGGTInterface

from ...infrastructure.constants import OperatorIDEnum

from .base_ot import MosplatOperatorBase, OperatorReturnItemsSet, OperatorPollReqs


class Mosplat_OT_initialize_model(MosplatOperatorBase):
    bl_idname = OperatorIDEnum.INITIALIZE_MODEL
    bl_description = (
        f"Install VGGT model weights from Hugging Face or load from cache if available."
    )

    poll_reqs = {OperatorPollReqs.PREFS, OperatorPollReqs.WINDOW_MANAGER}

    vggt_hf_id: StringProperty()  # pyright: ignore[reportInvalidTypeForm]
    vggt_outdir = StringProperty(
        subtype="DIR_PATH"
    )  # pyright: ignore[reportInvalidTypeForm]

    @classmethod
    def poll(cls, context) -> bool:
        if not super().poll(context):
            return False
        if MosplatVGGTInterface._initialized:
            cls.poll_message_set("Model has already been initialized.")
            return False  # prevent re-initialization
        return True

    def __init__(self, *args, **kwargs):
        """
        for modal operators, `__init__` is called before `invoke`
        docs: https://docs.blender.org/api/current/bpy.types.Operator.html#modal-execution
        """
        Operator.__init__(self, *args, **kwargs)

        self._queue = Queue()
        self._thread = None
        self._timer = None

    def modal(self, context, event) -> OperatorReturnItemsSet:
        if event.type != "TIMER":
            return {"RUNNING_MODAL", "PASS_THROUGH"}

        if not self._queue.empty():
            self._cleanup(context)

            _, payload = self._queue.get_nowait()

            if payload:
                self.logger().info("Successfully initialized VGGT model!")
                return {"FINISHED"}
            else:
                self.logger().error("VGGT model could not be initialized")
                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event) -> OperatorReturnItemsSet:
        prefs = self.prefs(context)
        self.vggt_hf_id = prefs.vggt_hf_id
        self.vggt_outdir = prefs.vggt_model_dir

        self._timer = self.wm(context).event_timer_add(
            time_step=0.1, window=context.window
        )

        return self.execute(context)

    def execute(self, context) -> OperatorReturnItemsSet:
        if not self.vggt_hf_id or not self.vggt_outdir:
            self.logger().error(
                "Call `invoke` to set up operator attributes before `execute`."
            )
            return {"CANCELLED"}

        vggt_outdir_path: Path = Path(self.vggt_outdir)  # convert to path here

        self._thread = threading.Thread(
            target=self._install_model_thread,
            args=(self.vggt_hf_id, vggt_outdir_path),
            daemon=True,
        )
        self._thread.start()

        self.wm(context).modal_handler_add(self)  # start timer polling here

        return {"RUNNING_MODAL"}

    def cancel(self, context):
        self._cleanup(context)

    def _install_model_thread(self, hf_id: str, outdir: Path):
        # put true or false initialize result in queue
        MosplatVGGTInterface.initialize_model(hf_id, outdir)

        """
        use initialization status rather than return result as `initialize_model`
        will return `False` if initialization status already occurred"""
        self._queue.put(("ok", MosplatVGGTInterface._initialized))

    def _cleanup(self, context: Context):
        if self._timer:
            self.wm(context).event_timer_remove(self._timer)
            self.logger().debug("Timer removed")
