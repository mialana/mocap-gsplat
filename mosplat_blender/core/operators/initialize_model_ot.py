from bpy.props import StringProperty

from pathlib import Path
import threading
from queue import Queue
from typing import Tuple

from ...interfaces import MosplatVGGTInterface

from ...infrastructure.schemas import OperatorIDEnum
from ...infrastructure.decorators import worker_fn_auto

from .base_ot import (
    MosplatOperatorBase,
    OperatorReturnItemsSet,
    OptionalOperatorReturnItemsSet,
)


class Mosplat_OT_initialize_model(MosplatOperatorBase[Tuple[str, bool]]):
    bl_idname = OperatorIDEnum.INITIALIZE_MODEL
    bl_description = (
        f"Install VGGT model weights from Hugging Face or load from cache if available."
    )

    vggt_hf_id: StringProperty(
        options={"SKIP_SAVE"}
    )  # pyright: ignore[reportInvalidTypeForm]
    vggt_outdir = StringProperty(
        subtype="DIR_PATH", options={"SKIP_SAVE"}
    )  # pyright: ignore[reportInvalidTypeForm]

    @classmethod
    def contexted_poll(cls, context, prefs, props) -> bool:
        if MosplatVGGTInterface._initialized:
            cls.poll_message_set("Model has already been initialized.")
            return False  # prevent re-initialization
        return True

    def queue_callback(self, context, event, next) -> OptionalOperatorReturnItemsSet:
        status, payload = next

        self.cleanup(context)

        if payload:
            self.logger().info("Successfully initialized VGGT model!")
            return {"FINISHED"}
        else:
            self.logger().error("VGGT model could not be initialized")
            return {"CANCELLED"}

    def contexted_execute(self, context) -> OperatorReturnItemsSet:
        prefs = self.prefs
        initialize_model_thread(
            self, hf_id=prefs.vggt_hf_id, model_cache_dir=prefs.vggt_model_dir
        )

        return {"RUNNING_MODAL"}


@worker_fn_auto
def initialize_model_thread(
    queue: Queue[Tuple[str, bool]],
    cancel_event: threading.Event,
    *,
    hf_id: str,
    model_cache_dir: Path,
):
    # put true or false initialize result in queue
    MosplatVGGTInterface.initialize_model(hf_id, model_cache_dir, cancel_event)

    """
    use initialization status rather than return result as `initialize_model`
    will return `False` if initialization status already occurred"""
    queue.put(("ok", MosplatVGGTInterface._initialized))
