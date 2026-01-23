from bpy.props import StringProperty

from pathlib import Path
import threading
from queue import Queue
from typing import Tuple

from ...interfaces import MosplatVGGTInterface

from ...infrastructure.constants import OperatorIDEnum, OperatorReturnItemsSet
from ...infrastructure.decorators import worker_fn_auto

from .base_ot import MosplatOperatorBase, OperatorPollReqs


class Mosplat_OT_initialize_model(MosplatOperatorBase[Tuple[str, bool]]):
    bl_idname = OperatorIDEnum.INITIALIZE_MODEL
    bl_description = (
        f"Install VGGT model weights from Hugging Face or load from cache if available."
    )

    __poll_reqs__ = {OperatorPollReqs.PREFS, OperatorPollReqs.WINDOW_MANAGER}

    vggt_hf_id: StringProperty(
        options={"SKIP_SAVE"}
    )  # pyright: ignore[reportInvalidTypeForm]
    vggt_outdir = StringProperty(
        subtype="DIR_PATH", options={"SKIP_SAVE"}
    )  # pyright: ignore[reportInvalidTypeForm]

    @classmethod
    def poll(cls, context) -> bool:
        if not super().poll(context):
            return False
        if MosplatVGGTInterface._initialized:
            cls.poll_message_set("Model has already been initialized.")
            return False  # prevent re-initialization
        return True

    def modal(self, context, event) -> OperatorReturnItemsSet:
        if event.type in {"RIGHTMOUSE", "ESC"}:
            self._cleanup(context)
            return {"CANCELLED"}
        elif event.type != "TIMER":
            return {"PASS_THROUGH"}

        while self._worker and (next := self._worker.dequeue()) is not None:
            status, payload = next

            self._cleanup(context)

            if payload:
                self.logger().info("Successfully initialized VGGT model!")
                return {"FINISHED"}
            else:
                self.logger().error("VGGT model could not be initialized")
                return {"CANCELLED"}

        return {"RUNNING_MODAL", "PASS_THROUGH"}

    def execute(self, context) -> OperatorReturnItemsSet:
        prefs = self.prefs(context)
        self.vggt_hf_id = prefs.vggt_hf_id
        self.vggt_outdir = prefs.vggt_model_dir

        vggt_outdir_path: Path = Path(self.vggt_outdir)  # convert to path here

        self._install_model_thread(context, self.vggt_hf_id, vggt_outdir_path)

        return {"RUNNING_MODAL"}

    @worker_fn_auto
    def _install_model_thread(
        self,
        queue: Queue,
        cancel_event: threading.Event,
        hf_id: str,
        outdir: Path,
    ):
        # put true or false initialize result in queue
        MosplatVGGTInterface.initialize_model(hf_id, outdir)

        """
        use initialization status rather than return result as `initialize_model`
        will return `False` if initialization status already occurred"""
        queue.put(("ok", MosplatVGGTInterface._initialized))
