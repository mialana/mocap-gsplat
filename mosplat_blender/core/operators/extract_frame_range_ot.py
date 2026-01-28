from __future__ import annotations

from typing import Tuple, List, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass

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
    MediaIODataset,
    ProcessedFrameRange,
)
from ..handlers import restore_dataset_from_json
from ...infrastructure.decorators import worker_fn_auto
from ...infrastructure.constants import PER_FRAME_DIRNAME, RAW_FRAME_DIRNAME
from ...infrastructure.macros import is_path_accessible

if TYPE_CHECKING:
    from cv2 import VideoCapture


@dataclass(frozen=True)
class ThreadKwargs:
    updated_media_files: List[Path]
    frame_range: Tuple[int, int]
    data_output_dirpath: Path
    dataset_as_dc: MediaIODataset


class Mosplat_OT_extract_frame_range(
    MosplatOperatorBase[str, ThreadKwargs],
):
    bl_idname = OperatorIDEnum.EXTRACT_FRAME_RANGE
    bl_description = "Extract a frame range from all media files in media directory."

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

            # try setting all the properties that are needed for the op
            self._data_output_dirpath: Path = props.data_output_dirpath(prefs)
            self._media_files: List[Path] = props.media_files(prefs)
            self._frame_range: Tuple[int, int] = props.current_frame_range
            return self.execute(context)
        except UserFacingError as e:
            self.logger().error(str(e))
            return {"CANCELLED"}

    def contexted_execute(self, context) -> OperatorReturnItemsSet:
        self.operator_thread(
            self,
            _kwargs=ThreadKwargs(
                updated_media_files=self._media_files,
                frame_range=self._frame_range,
                data_output_dirpath=self._data_output_dirpath,
                dataset_as_dc=self.dataset_as_dc,
            ),
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

    @staticmethod
    @worker_fn_auto
    def operator_thread(queue, cancel_event, *, _kwargs):
        import cv2

        start, end = _kwargs.frame_range
        caps: List[cv2.VideoCapture] = []

        try:
            for media in _kwargs.updated_media_files:
                cap = cv2.VideoCapture(str(media))
                if not cap.isOpened():
                    raise UserFacingError(f"Could not open media file: {media}")
                caps.append(cap)

            # create a new frame range with both limits at start
            new_frame_range = ProcessedFrameRange(start_frame=start, end_frame=start)
            _kwargs.dataset_as_dc.processed_frame_ranges.append(new_frame_range)

            for frame_idx in range(start, end):
                if cancel_event.is_set():
                    return
                frame_dir = _kwargs.data_output_dirpath.joinpath(
                    PER_FRAME_DIRNAME.format(frame_idx)
                )
                frame_dir.mkdir(parents=True, exist_ok=True)

                frame_npy_filepath = frame_dir.joinpath(f"{RAW_FRAME_DIRNAME}.npy")
                if not is_path_accessible(frame_npy_filepath):
                    _write_frame_data_to_npy(frame_idx, caps, frame_npy_filepath)

                new_frame_range.end_frame = frame_idx
                queue.put("update")
        except UserFacingError as e:
            queue.put(str(e))  # exit early wherever error occurs and put error on queue
        finally:
            for cap in caps:
                cap.release()

        queue.put("done")


def _write_frame_data_to_npy(frame_idx: int, caps: List[VideoCapture], out_path: Path):
    import cv2
    import numpy as np

    images: List[cv2.typing.MatLike] = []
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            raise UserFacingError(f"Failed to read frame: {frame_idx}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
        images.append(frame)

    stacked = np.stack(images, axis=0)
    np.save(out_path, stacked)
