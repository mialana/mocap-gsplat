from __future__ import annotations

import sys
from numbers import Integral
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, cast

from infrastructure.macros import is_path_accessible, save_tensor_stack_png_preview
from infrastructure.schemas import (
    FrameTensorMetadata,
    ImagesTensorType,
    MediaIOMetadata,
    ProcessedFrameRange,
    SavedTensorFileName,
)
from operators.base_ot import MosplatOperatorBase


class ProcessKwargs(NamedTuple):
    updated_media_files: List[Path]
    frame_range: Tuple[int, int]
    out_file_formatter: str
    data: MediaIOMetadata


class Mosplat_OT_extract_frame_range(
    MosplatOperatorBase[Tuple[str, str, Optional[MediaIOMetadata]], ProcessKwargs],
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
        self._frame_range: Tuple[int, int] = props.frame_range_
        self._out_file_formatter: str = props.generate_safetensor_filepath_formatters(
            prefs, [SavedTensorFileName.RAW]
        )[SavedTensorFileName.RAW]

        return self.execute_with_package(pkg)

    def _contexted_execute(self, pkg):
        self.launch_subprocess(
            pkg.context,
            pwargs=ProcessKwargs(
                updated_media_files=self._media_files,
                frame_range=self._frame_range,
                out_file_formatter=self._out_file_formatter,
                data=self.data,
            ),
        )

        return "RUNNING_MODAL"

    def _queue_callback(self, pkg, event, next):
        status, msg, new_data = next
        # sync props regardless as the updated dataclass is still valid
        if new_data:
            self.data = new_data
            self.sync_to_props(pkg.props)

        if status == "done":
            pkg.props.was_frame_range_extracted = True
        return super()._queue_callback(pkg, event, next)

    @staticmethod
    def _operator_subprocess(queue, cancel_event, *, pwargs):
        import torch
        from safetensors.torch import save_file
        from torchcodec.decoders import VideoDecoder

        files, (start, end), out_file_formatter, data = pwargs

        torchcodec_device: str = (
            "cuda"
            if not sys.platform == "win32" and torch.cuda.is_available()
            else "cpu"  # torchcodec does not ship cuda-enabled wheels through PyPI on Windows
        )

        decoders: List[VideoDecoder] = []
        try:
            for media_file in files:
                dec = VideoDecoder(media_file, device=torchcodec_device)
                decoders.append(dec)
        except ValueError as e:
            # exit early wherever error occurs and put error on queue
            queue.put(("error", str(e), None))
            return

        # create a new frame range with both limits at start
        new_frame_range = ProcessedFrameRange(start_frame=start, end_frame=start)
        data.add_frame_range(new_frame_range)

        for idx in range(start, end):
            if cancel_event.is_set():
                return

            out_file = Path(out_file_formatter.format(frame_idx=idx))

            if is_path_accessible(out_file):
                note = f"Safetensor data for frame '{idx}' already found on disk."
            else:
                out_file.parent.mkdir(parents=True, exist_ok=True)

                tensor_list = [dec[cast(Integral, idx)] for dec in decoders]
                # convert to 0.0-1.0 range
                tensor: ImagesTensorType = (
                    torch.stack(tensor_list, dim=0).float() / 255.0
                )

                save_file(
                    {SavedTensorFileName._default_tensor_key(): tensor},
                    filename=out_file,
                    metadata=FrameTensorMetadata(
                        frame_idx=idx, media_files=files
                    ).to_dict(),
                )
                save_tensor_stack_png_preview(tensor, out_file)

                note = f"Saved safetensor '{out_file}' to disk."

            new_frame_range.end_frame = idx
            queue.put((f"update", note, data))

        for dec in decoders:
            del dec  # release resources

        data.add_frame_range(new_frame_range)

        queue.put(("done", f"Frames '{start}-{end}' extracted.", None))


def process_entrypoint(*args, **kwargs):
    Mosplat_OT_extract_frame_range._operator_subprocess(*args, **kwargs)
