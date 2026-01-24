from pathlib import Path

import threading
from queue import Queue

from .base_ot import (
    MosplatOperatorBase,
    OperatorReturnItemsSet,
    OptionalOperatorReturnItemsSet,
)

from ...infrastructure.schemas import OperatorIDEnum, UserFacingError
from ...infrastructure.decorators import worker_fn_auto
from ...interfaces.media_io_interface import MediaProcessStatus


class Mosplat_OT_check_media_frame_counts(
    MosplatOperatorBase[tuple[str, MediaProcessStatus]]
):
    bl_idname = OperatorIDEnum.CHECK_MEDIA_FRAME_COUNTS
    bl_description = (
        "Check frame counts of all media files found in given media directory."
    )

    @classmethod
    def contexted_poll(cls, context, prefs, props) -> bool:
        try:
            return bool(
                prefs.media_extensions_set
                and props.data_output_dirpath
                and props.media_files
            )
        except UserFacingError as e:
            cls.poll_message_set(str(e))
            return False

    def contexted_execute(self, context) -> OperatorReturnItemsSet:
        self.props.metadata.media_process_statuses.clear()

        check_frame_counts_thread(self)

        return {"RUNNING_MODAL"}

    def queue_callback(self, context, event, next) -> OptionalOperatorReturnItemsSet:
        pass


@worker_fn_auto
def check_frame_counts_thread(
    queue: Queue,
    cancel_event: threading.Event,
):
    pass
