from pathlib import Path
import threading
from ...interfaces.vggt_interface import MosplatVGGTInterface

from ...infrastructure.constants import OperatorIDEnum

from .base import MosplatOperatorBase, OperatorReturnItemsSet


class Mosplat_OT_initialize_model(MosplatOperatorBase):
    bl_description = "Install VGGT model weights from Hugging Face."

    bl_idname = OperatorIDEnum.INITIALIZE_MODEL

    _thread = None

    _vggt_hf_id = ""
    _vggt_outdir = None

    def execute(self, context) -> OperatorReturnItemsSet:
        self._thread = threading.Thread(
            target=self._install_model_thread,
            args=(self._vggt_hf_id, self._vggt_outdir),
            daemon=True,
        )
        self._thread.start()

        return {"RUNNING_MODAL"}

    def invoke(self, context, event) -> OperatorReturnItemsSet:
        if not (prefs := self.prefs(context)):
            return {"CANCELLED"}

        self._vggt_hf_id = prefs.vggt_hf_id
        self._vggt_outdir = prefs.vggt_model_dir

        return self.execute(context)

    def _install_model_thread(self, hf_id: str, outdir: Path):
        MosplatVGGTInterface.initialize_model(hf_id, outdir)

        self.logger().debug("Install model thread completed!")
