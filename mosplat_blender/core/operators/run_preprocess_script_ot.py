from __future__ import annotations

from dataclasses import dataclass

from .base_ot import (
    MosplatOperatorBase,
    OperatorReturnItemsSet,
    OptionalOperatorReturnItemsSet,
)

from ...infrastructure.schemas import OperatorIDEnum, UserFacingError
from ..handlers import restore_dataset_from_json
from ...infrastructure.decorators import worker_fn_auto


@dataclass(frozen=True)
class ThreadKwargs:
    pass


class Mosplat_OT_run_preprocess_script(
    MosplatOperatorBase[str, ThreadKwargs],
):
    bl_idname = OperatorIDEnum.RUN_PREPROCESS_SCRIPT
    bl_description = "Run current preprocess script on current frame range."

    @classmethod
    def contexted_poll(cls, context, prefs, props) -> bool:
        if not props.dataset_accessor.is_valid_media_directory:
            cls._poll_error_msg_list.append(
                "Ensure that frame count, width, and height of all media files within current media directory match."
            )
        else:
            cls._poll_error_msg_list.extend(props.frame_range_err_list(prefs))

        return len(cls._poll_error_msg_list) == 0

    def contexted_invoke(self, context, event) -> OperatorReturnItemsSet:
        prefs = self.prefs
        props = self.props
        try:
            restore_dataset_from_json(props, prefs)  # try to restore from local JSON

            return self.execute(context)
        except UserFacingError as e:
            self.logger().error(str(e))
            return {"CANCELLED"}

    def contexted_execute(self, context) -> OperatorReturnItemsSet:

        return {"RUNNING_MODAL"}

    def queue_callback(self, context, event, next) -> OptionalOperatorReturnItemsSet:
        if next == "done":
            self.cleanup(context)  # write props (as dataclass) to JSON
            return

        if next != "update":  # if sent an error message via queue
            self.logger().warning(next)

        # sync props regardless as the updated dataclass is still valid
        self.props.dataset_accessor.from_dataclass(self.dataset_as_dc)

    @staticmethod
    @worker_fn_auto
    def operator_thread(queue, cancel_event, *, _kwargs):
        pass
