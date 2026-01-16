from bpy.props import StringProperty

from typing import List, ClassVar
from pathlib import Path

from ...infrastructure.constants import OperatorIDEnum

from .base_ot import MosplatOperatorBase, OperatorReturnItemsSet, OperatorPollReqs


class Mosplat_OT_run_inference(MosplatOperatorBase):
    bl_idname = OperatorIDEnum.RUN_INFERENCE
    bl_description = "Run inference on selected media directory."

    poll_reqs = {OperatorPollReqs.PROPS, OperatorPollReqs.WINDOW_MANAGER}

    _poll_error_msg_list: ClassVar[List[str]] = []

    @classmethod
    def poll(cls, context) -> bool:
        if not super().poll(context):
            return False

        cls._poll_error_msg_list.clear()

        cls._validate_media_directory(context)
        cls._validate_preferences(context)

        cls.poll_message_set("\n".join(cls._poll_error_msg_list))

        return not cls._poll_error_msg_list

    def execute(self, context) -> OperatorReturnItemsSet:

        return {"FINISHED"}

    @classmethod
    def _validate_media_directory(cls, context):
        pass

    @classmethod
    def _validate_preferences(cls, context):
        prefs = cls.prefs(context)

        # validate the script file preference
        preprocess_media_script_file = Path(prefs.preprocess_media_script_file)
        preprocess_media_script_file_name = prefs.bl_rna.properties[
            "preprocess_media_script_file"
        ].name

        if not preprocess_media_script_file.is_file():
            cls._poll_error_msg_list.append(
                f"'{preprocess_media_script_file_name}' is an invalid file"
            )

        if not preprocess_media_script_file.suffix == ".py":
            cls._poll_error_msg_list.append(
                f"'{preprocess_media_script_file_name}' is not a Python script"
            )

        if not prefs.data_output_dir:
            cls._poll_error_msg_list.append(
                f"'{prefs.bl_rna.properties['data_output_dir'].name}' is invalid."
            )

        return
