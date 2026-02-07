from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

from infrastructure.macros import is_path_accessible, write_frame_data_to_npz
from infrastructure.schemas import (
    FrameNPZStructure,
    MediaIODataset,
    ProcessedFrameRange,
    SavedNPZName,
    UserFacingError,
)
from operators.base_ot import MosplatOperatorBase


class ProcessKwargs(NamedTuple):
    updated_media_files: List[Path]
    frame_range: Tuple[int, int]
    npz_files: List[Path]
    dataset_as_dc: MediaIODataset


class Mosplat_OT_extract_frame_range(
    MosplatOperatorBase[Tuple[str, Optional[MediaIODataset]], ProcessKwargs],
):
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
        self._npz_files: List[Path] = props.generate_npz_filepaths_for_frame_range(
            prefs, [SavedNPZName.RAW], [False]
        )[SavedNPZName.RAW]

        return self.execute_with_package(pkg)

    def _contexted_execute(self, pkg):
        self.launch_subprocess(
            pkg.context,
            pwargs=ProcessKwargs(
                updated_media_files=self._media_files,
                frame_range=self._frame_range,
                npz_files=self._npz_files,
                dataset_as_dc=self.data,
            ),
        )

        return "RUNNING_MODAL"

    def _queue_callback(self, pkg, event, next):
        status, new_data = next
        if status == "done":
            self.cleanup(pkg)  # write props (as dataclass) to JSON
            return "FINISHED"
        elif status.startswith("update"):  # if sent an error message via queue
            self.logger.debug(next)
        else:
            self.logger.error(next)
            return "FINISHED"

        # sync props regardless as the updated dataclass is still valid
        if new_data:
            self.data = new_data
            self._sync_to_props(pkg.props)
        return "RUNNING_MODAL"

    @staticmethod
    def _operator_subprocess(queue, cancel_event, *, pwargs):
        import cv2
        import numpy as np

        start, _ = pwargs.frame_range
        caps: List[cv2.VideoCapture] = []

        data = pwargs.dataset_as_dc
        files = pwargs.updated_media_files

        try:
            for media in pwargs.updated_media_files:
                cap = cv2.VideoCapture(str(media))
                if not cap.isOpened():
                    raise UserFacingError(f"Could not open media file: {media}")
                caps.append(cap)

            # create a new frame range with both limits at start
            new_frame_range = ProcessedFrameRange(start_frame=start, end_frame=start)
            data.processed_frame_ranges.append(new_frame_range)

            for idx, raw_npz_file in enumerate(pwargs.npz_files):
                if cancel_event.is_set():
                    return

                if not is_path_accessible(raw_npz_file):
                    raw_npz_file.parent.mkdir(parents=True, exist_ok=True)
                    structure: FrameNPZStructure = {
                        "frame": np.array([idx], dtype=np.int32),
                        "media_files": np.array([str(f) for f in files], dtype=np.str_),
                    }
                    note = write_frame_data_to_npz(idx, caps, raw_npz_file, **structure)
                else:
                    note = f"NPZ data for frame index '{idx}' already found on disk. Skipping..."

                new_frame_range.end_frame = idx
                queue.put((f"update: {note}", data))
        except (UserFacingError, OSError) as e:
            # exit early wherever error occurs and put error on queue
            queue.put((str(e), None))
        finally:
            for cap in caps:
                cap.release()

        queue.put(("done", None))


def process_entrypoint(*args, **kwargs):
    Mosplat_OT_extract_frame_range._operator_subprocess(*args, **kwargs)
