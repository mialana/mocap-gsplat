from pathlib import Path

import threading
from queue import Queue

from .base_ot import (
    MosplatOperatorBase,
    OperatorReturnItemsSet,
    OptionalOperatorReturnItemsSet,
)

from ...infrastructure.schemas import (
    OperatorIDEnum,
    UserFacingError,
    DeveloperError,
    MediaIOMetadata,
)
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

        check_frame_counts_thread(self, metadata=self.metadata)

        return {"RUNNING_MODAL"}

    def queue_callback(self, context, event, next) -> OptionalOperatorReturnItemsSet:
        pass


@worker_fn_auto
def check_frame_counts_thread(
    queue: Queue, cancel_event: threading.Event, *, metadata: MediaIOMetadata
):
    pass


def _extract_media_frame_count(status: MediaProcessStatus):
    """use opencv to get the frame count of media"""
    import cv2

    def _cleanup(method: str):
        cap.release()
        status.frame_count = frame_count
        status.message = f"Read media file '{status.filepath}' with the frame count '{frame_count}' ({method})."

    cap = cv2.VideoCapture(status.filepath)
    if not cap.isOpened():
        raise DeveloperError(f"Could not open media file: {status.filepath}")

    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)  # seek to end
    duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps > 0 and duration_ms > 0:
        frame_count = int(round((duration_ms / 1000.0) * fps))
        if frame_count > 0:
            return _cleanup("fps + duration metadata")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if 0 < frame_count < 2**32 - 1:
        return _cleanup("frame count metadata")

    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.0)  # return seek to start

    frame_count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        frame_count += 1

    return _cleanup("manual")
