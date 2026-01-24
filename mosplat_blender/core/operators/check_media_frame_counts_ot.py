from typing import Tuple, Set, List
import threading
from queue import Queue
from pathlib import Path

from bpy.types import Context, Event

from .base_ot import (
    MosplatOperatorBase,
    OperatorReturnItemsSet,
    OptionalOperatorReturnItemsSet,
)

from ...infrastructure.schemas import (
    OperatorIDEnum,
    UserFacingError,
    MediaIOMetadata,
    MediaProcessStatus,
)
from ...infrastructure.decorators import worker_fn_auto
from ..checks import (
    check_media_extensions,
    check_data_output_dirpath,
    check_media_files,
)
from ..handlers import restore_metadata_from_json


class Mosplat_OT_check_media_frame_counts(MosplatOperatorBase[Tuple[str]]):
    bl_idname = OperatorIDEnum.CHECK_MEDIA_FRAME_COUNTS
    bl_description = (
        "Check frame counts of all media files found in given media directory."
    )

    def contexted_invoke(
        self, context: Context, event: Event
    ) -> OperatorReturnItemsSet:
        prefs = self.prefs
        props = self.props
        try:
            restore_metadata_from_json(props, prefs)  # try to restore from local JSON

            # try setting all the properties that are needed for the op
            prefs.media_extensions_set = check_media_extensions(prefs)
            props.data_output_dirpath = check_data_output_dirpath(prefs, props)
            props.media_files = check_media_files(prefs, props)
            return self.execute(context)
        except UserFacingError as e:
            self.logger().error(str(e))
            return {"CANCELLED"}

    def contexted_execute(self, context) -> OperatorReturnItemsSet:
        if self.props.media_files is None:
            return {"CANCELLED"}

        check_frame_counts_thread(
            self,
            updated_media_files=self.props.media_files,
            metadata_dc=self.metadata_dc,
        )

        return {"RUNNING_MODAL"}

    def queue_callback(self, context, event, next) -> OptionalOperatorReturnItemsSet:
        if next == "update":  # keep props up-to-date so updates are reflected in UI
            self.props.metadata_ptr.from_dataclass(self.metadata_dc)
        elif next == "done":
            self.cleanup(context)  # write props (as dataclass) to JSON


@worker_fn_auto
def check_frame_counts_thread(
    queue: Queue[str],
    cancel_event: threading.Event,
    *,
    updated_media_files: Set[Path],
    metadata_dc: MediaIOMetadata,
):
    status_lookup = MediaProcessStatus.as_lookup(metadata_dc.media_process_statuses)

    # replace the statuses with the fresh validated list we are building
    updated_statuses: List[MediaProcessStatus] = []
    metadata_dc.media_process_statuses = updated_statuses

    for file in updated_media_files:
        if cancel_event.is_set():
            return  # simply return as new queue items will not be read anymore

        # check if the status for the file already exists, create new if not
        status = status_lookup.get(str(file), MediaProcessStatus(filepath=str(file)))

        # if success is already valid skip new extraction
        success = status.is_valid or _extract_media_frame_count(
            status
        )  # overwrite `success` here

        target_frame_count = max(
            metadata_dc.collective_media_frame_count, status.frame_count
        )  # weed out if either are -1

        metadata_dc.do_media_durations_all_match = (
            metadata_dc.do_media_durations_all_match
            and success
            and target_frame_count == status.frame_count
        )  # accumulate success

        metadata_dc.collective_media_frame_count = (
            target_frame_count if metadata_dc.do_media_durations_all_match else -1
        )

        updated_statuses.append(status)

        queue.put("update")  # transmit that metadata has been updated

    queue.put("done")  # signal done
    return


def _extract_media_frame_count(status: MediaProcessStatus) -> bool:
    """use opencv to get the frame count of media"""
    import cv2

    def _cleanup(method: str):
        cap.release()
        stat = Path(status.filepath).stat()

        status.overwrite(
            is_valid=True,
            frame_count=frame_count,
            message=f"Read file with the frame count '{frame_count}' ({method}).",
            mod_time=stat.st_mtime,
            file_size=stat.st_size,
        )
        return True

    cap = cv2.VideoCapture(status.filepath)
    if not cap.isOpened():
        status.is_valid = False
        status.message = f"Could not open file."
        return False  # mark invalid and update message

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
