from __future__ import annotations

from typing import Tuple, List, TYPE_CHECKING
import threading
from queue import Queue
from pathlib import Path

from .base_ot import (
    MosplatOperatorBase,
    OperatorReturnItemsSet,
    OptionalOperatorReturnItemsSet,
)
from ..handlers import restore_dataset_from_json

from ...infrastructure.schemas import (
    OperatorIDEnum,
    UserFacingError,
    MediaIODataset,
    MediaFileStatus,
)
from ...infrastructure.decorators import worker_fn_auto

if TYPE_CHECKING:
    from cv2 import VideoCapture


class Mosplat_OT_validate_media_file_statuses(MosplatOperatorBase[Tuple[str]]):
    bl_idname = OperatorIDEnum.VALIDATE_MEDIA_FILE_STATUSES
    bl_description = "Check frame count, width, and height of all media files found in current media directory."

    def contexted_invoke(self, context, event) -> OperatorReturnItemsSet:
        prefs = self.prefs
        props = self.props
        try:
            restore_dataset_from_json(props, prefs)  # try to restore from local JSON

            # try setting all the properties that are needed for the op
            self._media_files: List[Path] = props.media_files(prefs)
            return self.execute(context)
        except UserFacingError as e:
            self.logger().error(str(e))
            return {"CANCELLED"}

    def contexted_execute(self, context) -> OperatorReturnItemsSet:
        validate_common_media_features_thread(
            self,
            updated_media_files=self._media_files,
            dataset_as_dc=self.dataset_as_dc,
        )

        return {"RUNNING_MODAL"}

    def queue_callback(self, context, event, next) -> OptionalOperatorReturnItemsSet:
        if next == "done":
            self.cleanup(context)  # write props (as dataclass) to JSON
            return

        if next != "update":  # if sent an error message via queue
            self.logger().warning(next)

        # sync props regardless as the updated dataclass is still valid
        self.props.dataset_accessor.from_dataclass(self.dataset_as_dc)


@worker_fn_auto
def validate_common_media_features_thread(
    queue: Queue[str],
    cancel_event: threading.Event,
    *,
    updated_media_files: List[Path],
    dataset_as_dc: MediaIODataset,
):
    status_lookup = MediaFileStatus.as_lookup(dataset_as_dc.media_file_statuses)

    # replace the statuses with the fresh validated list we are building
    updated_statuses: List[MediaFileStatus] = []
    dataset_as_dc.media_file_statuses = updated_statuses

    for file in updated_media_files:
        if cancel_event.is_set():
            return  # simply return as new queue items will not be read anymore
        queue_msg = "update"

        # check if the status for the file already exists, create new if not
        status = status_lookup.get(str(file), MediaFileStatus(filepath=str(file)))

        # if success is already valid skip new extraction
        if not status.is_valid:
            try:
                _extract_media_status(status)
            except UserFacingError as e:
                queue_msg = str(e)  # change queue message to error message

        dataset_as_dc.accumulate_media_status(status)  # update dataset from new status

        updated_statuses.append(status)

        queue.put(queue_msg)  # transmit that dataset has been updated

    queue.put("done")  # signal done
    return


def _extract_media_status(status: MediaFileStatus):
    """use opencv to get width, height, and frame count of media"""
    import cv2

    cap = cv2.VideoCapture(status.filepath)
    if not cap.isOpened():
        status.mark_invalid()
        raise UserFacingError(f"Could not open media file: {status.filepath}")

    stat = Path(status.filepath).stat()

    status.overwrite(
        is_valid=True,
        frame_count=_extract_media_frame_count(cap),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        mod_time=stat.st_mtime,
        file_size=stat.st_size,
    )

    cap.release()
    return


def _extract_media_frame_count(cap: VideoCapture) -> int:
    import cv2

    frame_count = -1
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)  # seek to end
    duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps > 0 and duration_ms > 0:
        estimated = int(round((duration_ms / 1000.0) * fps))
        if estimated > 0:
            frame_count = estimated
    else:
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if 0 < reported < 2**32 - 1:
            frame_count = reported
        else:
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.0)  # return seek to start

            count = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                count += 1

            frame_count = count

    return frame_count
