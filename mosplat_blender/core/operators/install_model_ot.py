from typing import TYPE_CHECKING, Set, TypeAlias
from pathlib import Path

from .base import MosplatOperatorBase

if TYPE_CHECKING:
    from bpy.stub_internal.rna_enums import OperatorReturnItems
else:
    OperatorReturnItems: TypeAlias = str


class Mosplat_OT_install_model(MosplatOperatorBase):
    """install VGGT model weights from HuggingFace."""

    short_name = "install_model"

    _thread = None

    def execute(self, context) -> Set[OperatorReturnItems]:
        # self._thread = threading.Thread(
        #     target=self._install_model_thread, args=(context,), daemon=True
        # )
        # self._thread.start()

        return {"RUNNING_MODAL"}

    def _install_model_thread(self, installation_path: Path):
        pass

    def progress_bar(self, context):
        if not self.layout:
            return
        row = self.layout.row()
        row.progress(
            factor=context.window_manager.progress,
            type="BAR",
            text=(
                "Operation in progress..."
                if context.window_manager.progress < 1
                else "Operation Finished !"
            ),
        )
