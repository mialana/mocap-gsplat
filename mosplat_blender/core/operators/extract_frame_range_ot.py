from typing import Tuple, List
import threading
from queue import Queue
from pathlib import Path

from .base_ot import (
    MosplatOperatorBase,
    OperatorReturnItemsSet,
    OptionalOperatorReturnItemsSet,
)

from ..preferences import Mosplat_AP_Global
from ..properties import Mosplat_PG_Global

from ...infrastructure.schemas import (
    OperatorIDEnum,
    UserFacingError,
    MediaIOMetadata,
    ProcessedFrameRange,
)
from ...infrastructure.decorators import worker_fn_auto
from ..handlers import restore_metadata_from_json

FRAME_DIR_FMT = "frame_{:04d}"
RAW_DIR_NAME = "raw"


class Mosplat_OT_extract_frame_range(MosplatOperatorBase[Tuple[str]]):
    bl_idname = OperatorIDEnum.EXTRACT_FRAME_RANGE
    bl_description = "Extract a frame range from all media files in media directory."

    @classmethod
    def contexted_poll(cls, context, prefs, props) -> bool:
        cls._validate_frame_range(prefs, props)

        return len(cls._poll_error_msg_list) == 0

    @classmethod
    def _validate_frame_range(cls, prefs: Mosplat_AP_Global, props: Mosplat_PG_Global):
        start, end = props.current_frame_range
        prop_name = props.get_prop_name("current_frame_range")

        if start >= end:
            cls._poll_error_msg_list.append(
                f"Start frame for '{prop_name}' must be less than end frame."
            )

        if end >= props.metadata_ptr.common_frame_count:
            cls._poll_error_msg_list.append(
                f"End frame must be less than '{props.metadata_ptr.get_prop_name('common_frame_count')}' of '{props.metadata_ptr.common_frame_count}' frames."
            )

        max_frame_range = prefs.max_frame_range
        max_frame_range_name = prefs.get_prop_name("max_frame_range")
        if max_frame_range != -1 and end - start > prefs.max_frame_range:
            cls._poll_error_msg_list.append(
                f"For best results, set '{prop_name}' to less than '{max_frame_range}'.\n"
                f"Customize this restriction in the addon's preferences under '{max_frame_range_name}'"
            )

        return

    def contexted_invoke(self, context, event) -> OperatorReturnItemsSet:
        prefs = self.prefs
        props = self.props
        try:
            restore_metadata_from_json(props, prefs)  # try to restore from local JSON

            # try setting all the properties that are needed for the op
            self._media_files: List[Path] = props.media_files(prefs)
            self._frame_range: Tuple[int, int] = props.current_frame_range
            return self.execute(context)
        except UserFacingError as e:
            self.logger().error(str(e))
            return {"CANCELLED"}

    def contexted_execute(self, context) -> OperatorReturnItemsSet:
        extract_frame_range_thread(
            self,
            updated_media_files=self._media_files,
            frame_range=self._frame_range,
            metadata_dc=self.metadata_dc,
        )

        return {"RUNNING_MODAL"}

    def queue_callback(self, context, event, next) -> OptionalOperatorReturnItemsSet:
        if next == "update":  # keep props up-to-date so updates are reflected in UI
            self.props.metadata_ptr.from_dataclass(self.metadata_dc)
        elif next == "done":
            self.cleanup(context)  # write props (as dataclass) to JSON


@worker_fn_auto
def extract_frame_range_thread(
    queue: Queue[str],
    cancel_event: threading.Event,
    *,
    updated_media_files: List[Path],
    frame_range: Tuple[int, int],
    metadata_dc: MediaIOMetadata,
):
    start, end = frame_range
    base_dir = Path(metadata_dc.base_directory)

    for frame_idx in range(start, end):
        if cancel_event.is_set():
            return

        frame_dir = base_dir / FRAME_DIR_FMT
        frame_dir.mkdir(parents=True, exist_ok=True)

        _extract_frame_to_npy(
            frame_idx=frame_idx,
            media_files=updated_media_files,
            out_dir=frame_dir,
            stage_name="raw",
        )

        queue.put("update")

    metadata_dc.processed_frame_ranges.append(
        ProcessedFrameRange(
            start_frame=start,
            end_frame=end,
        )
    )

    queue.put("done")


def _extract_frame_to_npy(
    *,
    frame_idx: int,
    media_files: List[Path],
    out_dir: Path,
    stage_name: str,
):
    import cv2
    import numpy as np

    caps = []
    try:
        for media in media_files:
            cap = cv2.VideoCapture(str(media))
            if not cap.isOpened():
                raise UserFacingError(f"Could not open media file: {media}")
            caps.append(cap)

        images = []
        for cap in caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                raise UserFacingError(f"Failed to read frame {frame_idx}")
            # BGR → RGB once, here
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(frame)

        stacked = np.stack(images, axis=0)
        np.save(out_dir / f"{stage_name}.npy", stacked)

    finally:
        for cap in caps:
            cap.release()
