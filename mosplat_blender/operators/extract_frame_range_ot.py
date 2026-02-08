from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, cast

from infrastructure.macros import is_path_accessible
from infrastructure.schemas import (
    FrameTensorMetadata,
    MediaIODataset,
    ProcessedFrameRange,
    SavedTensorFileName,
    UserFacingError,
)
from operators.base_ot import MosplatOperatorBase


class ProcessKwargs(NamedTuple):
    updated_media_files: List[Path]
    frame_range: Tuple[int, int]
    st_files: List[Path]
    dataset_as_dc: MediaIODataset


class Mosplat_OT_extract_frame_range(
    MosplatOperatorBase[Tuple[str, str, Optional[MediaIODataset]], ProcessKwargs],
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
        self._frame_range: Tuple[int, int] = tuple(props.current_frame_range)
        self._st_files: List[Path] = props.generate_st_filepaths_for_frame_range(
            prefs, [SavedTensorFileName.RAW], [False]
        )[SavedTensorFileName.RAW]

        return self.execute_with_package(pkg)

    def _contexted_execute(self, pkg):
        self.launch_subprocess(
            pkg.context,
            pwargs=ProcessKwargs(
                updated_media_files=self._media_files,
                frame_range=self._frame_range,
                st_files=self._st_files,
                dataset_as_dc=self.data,
            ),
        )

        return "RUNNING_MODAL"

    def _queue_callback(self, pkg, event, next):
        status, msg, new_data = next
        # sync props regardless as the updated dataclass is still valid
        if new_data:
            self.data = new_data
            self._sync_to_props(pkg.props)
        return super()._queue_callback(pkg, event, next)

    @staticmethod
    def _operator_subprocess(queue, cancel_event, *, pwargs):
        import torch
        from safetensors.torch import save_file
        from torchcodec.decoders import VideoDecoder

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        start, end = pwargs.frame_range
        decoders: List[VideoDecoder] = []

        data = pwargs.dataset_as_dc
        files = pwargs.updated_media_files

        try:
            for media_file in pwargs.updated_media_files:
                dec = VideoDecoder(media_file, device=device)
                decoders.append(dec)

            # create a new frame range with both limits at start
            new_frame_range = ProcessedFrameRange(start_frame=start, end_frame=start)
            data.processed_frame_ranges.append(new_frame_range)

            for idx, out_file in enumerate(pwargs.st_files):
                if cancel_event.is_set():
                    return

                if is_path_accessible(out_file):
                    note = f"Safetensor data for frame '{idx}' already found on disk."
                else:
                    out_file.parent.mkdir(parents=True, exist_ok=True)

                    tensor_list = [dec[cast(Integral, idx)] for dec in decoders]
                    tensor = torch.stack(tensor_list, dim=0)

                    save_file(
                        {"data": tensor},
                        filename=out_file,
                        metadata=FrameTensorMetadata(
                            frame=idx, media_files=files
                        ).to_dict(),
                    )

                    note = f"Saved safetensor '{out_file}' to disk."

                new_frame_range.end_frame = idx
                queue.put((f"update", note, data))
        except (UserFacingError, OSError) as e:
            # exit early wherever error occurs and put error on queue
            queue.put(("error", str(e), None))
        finally:
            for dec in decoders:
                del dec  # release resources

        queue.put(("done", f"Frames '{start}-{end}' extracted.", None))


def process_entrypoint(*args, **kwargs):
    Mosplat_OT_extract_frame_range._operator_subprocess(*args, **kwargs)
