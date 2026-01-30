from __future__ import annotations

from typing import Tuple, List, Generator, NamedTuple
from pathlib import Path

from .base_ot import MosplatOperatorBase

from ...infrastructure.schemas import (
    UserFacingError,
    OperatorIDEnum,
    MediaIODataset,
    ProcessedFrameRange,
)
from ...infrastructure.macros import is_path_accessible, write_frame_data_to_npy


class ThreadKwargs(NamedTuple):
    updated_media_files: List[Path]
    frame_range: Tuple[int, int]
    npy_fp_generator: Generator[Path]
    dataset_as_dc: MediaIODataset


class Mosplat_OT_extract_frame_range(
    MosplatOperatorBase[str, ThreadKwargs],
):
    bl_idname = OperatorIDEnum.EXTRACT_FRAME_RANGE
    bl_description = "Extract a frame range from all media files in media directory."

    @classmethod
    def _contexted_poll(cls, pkg):
        props = pkg.props
        cls._poll_error_msg_list.extend(props.is_valid_media_directory_poll_result)
        cls._poll_error_msg_list.extend(props.frame_range_poll_result(pkg.prefs))

        return len(cls._poll_error_msg_list) == 0

    def _contexted_invoke(self, pkg, event):
        prefs = pkg.prefs
        props = pkg.props

        # try setting all the properties that are needed for the op
        self._media_files: List[Path] = props.media_files(prefs)
        self._frame_range: Tuple[int, int] = props.current_frame_range
        self._npy_filepath_generator: Generator[Path] = (
            props.generate_frame_range_npy_filepaths(prefs)
        )

        return self.execute_with_package(pkg)

    def _contexted_execute(self, pkg):
        self.launch_thread(
            pkg.context,
            twargs=ThreadKwargs(
                updated_media_files=self._media_files,
                frame_range=self._frame_range,
                npy_fp_generator=self._npy_filepath_generator,
                dataset_as_dc=self.data,
            ),
        )

        return "RUNNING_MODAL"

    def _queue_callback(self, pkg, event, next):
        if next == "done":
            self.cleanup(pkg)  # write props (as dataclass) to JSON
            return "FINISHED"

        if next != "update":  # if sent an error message via queue
            self.warn(next)

        # sync props regardless as the updated dataclass is still valid
        pkg.props.dataset_accessor.from_dataclass(self.data)
        return "RUNNING_MODAL"

    @staticmethod
    def _operator_thread(queue, cancel_event, *, twargs):
        import cv2

        start, _ = twargs.frame_range
        caps: List[cv2.VideoCapture] = []

        try:
            for media in twargs.updated_media_files:
                cap = cv2.VideoCapture(str(media))
                if not cap.isOpened():
                    raise UserFacingError(f"Could not open media file: {media}")
                caps.append(cap)

            # create a new frame range with both limits at start
            new_frame_range = ProcessedFrameRange(start_frame=start, end_frame=start)
            twargs.dataset_as_dc.processed_frame_ranges.append(new_frame_range)

            for idx, npy_filepath in enumerate(twargs.npy_fp_generator):
                if cancel_event.is_set():
                    return

                npy_filepath.parent.mkdir(parents=True, exist_ok=True)

                if not is_path_accessible(npy_filepath):
                    write_frame_data_to_npy(idx, caps, npy_filepath)

                new_frame_range.end_frame = idx
                queue.put("update")
        except (UserFacingError, OSError) as e:
            queue.put(str(e))  # exit early wherever error occurs and put error on queue
        finally:
            for cap in caps:
                cap.release()

        queue.put("done")
