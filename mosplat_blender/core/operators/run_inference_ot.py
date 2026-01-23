import bpy
from typing import List, ClassVar
from pathlib import Path

from ...infrastructure.schemas import OperatorIDEnum

from .base_ot import MosplatOperatorBase, OperatorReturnItemsSet
from ..preferences import Mosplat_AP_Global
from ..properties import Mosplat_PG_Global


class Mosplat_OT_run_inference(MosplatOperatorBase):
    bl_idname = OperatorIDEnum.RUN_INFERENCE
    bl_description = "Run inference on selected media directory."

    _poll_error_msg_list: ClassVar[List[str]] = []

    @classmethod
    def contexted_poll(cls, context, prefs, props) -> bool:
        cls._poll_error_msg_list.clear()

        cls._validate_frame_range(prefs, props)
        cls._validate_preprocess_media_script(prefs)

        cls.poll_message_set("\n".join(cls._poll_error_msg_list))

        return not cls._poll_error_msg_list

    def contexted_execute(self, context) -> OperatorReturnItemsSet:
        return {"FINISHED"}

    @classmethod
    def _validate_frame_range(cls, prefs: Mosplat_AP_Global, props: Mosplat_PG_Global):
        start, end = props.current_frame_range
        prop_name = props.get_prop_name("current_frame_range")

        if start >= end:
            cls._poll_error_msg_list.append(
                f"Start component for '{prop_name}' must be less than end component."
            )

        max_frame_range = prefs.max_frame_range
        max_frame_range_name = prefs.get_prop_name("max_frame_range")
        if max_frame_range != -1 and end - start > prefs.max_frame_range:
            cls._poll_error_msg_list.append(
                f"For best results, set '{prop_name}' to less than '{max_frame_range}'.\n"
                f"Customize this restriction in the addon's preferences under '{max_frame_range_name}'"
            )
        return

    @classmethod
    def _validate_preprocess_media_script(cls, prefs: Mosplat_AP_Global):
        # validate the current selected script file preference
        target_pref = Path(prefs.preprocess_media_script_file)
        target_pref_name = prefs.get_prop_name("preprocess_media_script_file")

        if not target_pref.is_file():
            cls._poll_error_msg_list.append(f"'{target_pref_name}' is an invalid file.")

        if not target_pref.suffix == ".py":
            cls._poll_error_msg_list.append(
                f"'{target_pref_name}' is not a Python script."
            )

        return
