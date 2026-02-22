from __future__ import annotations

import sys
from functools import partial
from numbers import Integral
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, cast

from ..infrastructure.macros import (
    crop_tensor,
    load_and_verify_tensor_file,
    save_images_tensor,
    save_tensor_stack_png_preview,
    to_0_1,
)
from ..infrastructure.schemas import (
    CropGeometry,
    FrameTensorMetadata,
    MediaIOMetadata,
    ProcessedFrameRange,
    SavedTensorFileName,
    SavedTensorKey,
    TensorTypes as TT,
    UserAssertionError,
    UserFacingError,
)
from .base_ot import MosplatOperatorBase


class ProcessKwargs(NamedTuple):
    updated_media_files: List[Path]
    frame_range: Tuple[int, int]
    exported_file_formatter: str
    create_preview_images: bool
    median_height: int
    median_width: int
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
        self._exported_file_formatter: str = props.exported_file_formatter(prefs)

        return self.execute_with_package(pkg)

    def _contexted_execute(self, pkg):
        accessor = pkg.props.metadata_accessor

        self.launch_subprocess(
            pkg.context,
            pwargs=ProcessKwargs(
                updated_media_files=self._media_files,
                frame_range=self._frame_range,
                exported_file_formatter=self._exported_file_formatter,
                median_height=int(accessor.median_height),
                median_width=int(accessor.median_width),
                create_preview_images=bool(pkg.prefs.create_preview_images),
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
        from torchcodec.decoders import VideoDecoder

        files, (start, end), exported_file_formatter, preview, H, W, data = pwargs

        out_file_formatter = partial(
            exported_file_formatter.format,
            file_name=SavedTensorFileName.RAW,
            file_ext="safetensors",
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torchcodec_device = (
            torch.device("cpu") if sys.platform == "win32" else device
        )  # torchcodec does not ship cuda-enabled wheels through PyPI on Windows

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

        crop_geom = CropGeometry.from_image_dims(H=H, W=W)

        for idx in range(start, end):
            if cancel_event.is_set():
                return

            out_file = Path(out_file_formatter(frame_idx=idx))
            new_metadata = FrameTensorMetadata(
                frame_idx=idx,
                media_files=files,
                preprocess_script=None,
                model_options=None,
            )

            try:
                _ = load_and_verify_tensor_file(
                    out_file,
                    device,
                    new_metadata,
                    keys=[SavedTensorKey.IMAGES],
                )
                note = f"Safetensor data for frame '{idx}' already found on disk."
            except (OSError, UserAssertionError, UserFacingError):
                out_file.parent.mkdir(parents=True, exist_ok=True)

                tensor_list = [dec[cast(Integral, idx)].to(device) for dec in decoders]

                tensor_0_1: TT.ImagesTensor_0_1 = crop_tensor(
                    to_0_1(torch.stack(tensor_list, dim=0)), crop_geom
                )  # crop tensors as soon as possible

                save_images_tensor(out_file, new_metadata, tensor_0_1, None)

                if preview:
                    save_tensor_stack_png_preview(tensor_0_1, out_file)

                note = f"Saved safetensor '{out_file}' to disk."

            new_frame_range.end_frame = idx
            queue.put((f"update", note, data))

        for dec in decoders:
            del dec  # release resources

        data.add_frame_range(new_frame_range)

        queue.put(
            (
                "done",
                f"Frames '{start}-{end}' extracted and images were processed to width of '{crop_geom.new_W}' and height of '{crop_geom.new_H}'.",
                None,
            )
        )


def process_entrypoint(*args, **kwargs):
    Mosplat_OT_extract_frame_range._operator_subprocess(*args, **kwargs)
