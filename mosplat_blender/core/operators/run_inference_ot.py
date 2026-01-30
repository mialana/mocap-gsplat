from typing import List, ClassVar

from ...infrastructure.schemas import OperatorIDEnum

from .base_ot import MosplatOperatorBase


class Mosplat_OT_run_inference(MosplatOperatorBase):
    bl_idname = OperatorIDEnum.RUN_INFERENCE
    bl_description = "Run inference on selected media directory."

    _poll_error_msg_list: ClassVar[List[str]] = []

    def contexted_execute(self, pkg):
        return "FINISHED"
