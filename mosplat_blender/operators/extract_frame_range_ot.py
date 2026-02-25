from __future__ import annotations

import sys
from numbers import Integral
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, cast

from ..infrastructure.schemas import (
    ExportedFileName,
    FrameTensorMetadata,
    MediaIOMetadata,
    ProcessedFrameRange,
    UserAssertionError,
    UserFacingError,
)
from .base_ot import MosplatOperatorBase


class ProcessKwargs(NamedTuple):
    updated_media_files: List[Path]
    frame_range: Tuple[int, int]
    exported_file_formatter: str
    create_preview_images: bool
    force: bool
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
        props = pkg.props
        prefs = pkg.prefs

        self.launch_subprocess(
            pkg.context,
            pwargs=ProcessKwargs(
                updated_media_files=self._media_files,
                frame_range=self._frame_range,
                exported_file_formatter=self._exported_file_formatter,
                create_preview_images=bool(prefs.create_preview_images),
                force=bool(prefs.force_all_operations),
                data=self.data,
            ),
        )

        return "RUNNING_MODAL"

    def _queue_callback(self, pkg, event, next):
        status, msg, new_data = next
        props = pkg.props

        # sync props regardless as the updated dataclass is still valid
        if new_data:
            self.data = new_data
            self.sync_to_props(props)

        if status == "done":
            props.was_frame_range_extracted = True
            props.was_frame_range_preprocessed = False
            props.ran_inference_on_frame_range = False
        return super()._queue_callback(pkg, event, next)

    @staticmethod
    def _operator_subprocess(queue, cancel_event, *, pwargs):
        import torch
        from torchcodec.decoders import VideoDecoder

        from ..infrastructure.dl_ops import (
            TensorTypes as TensorTypes,
            load_safetensors,
            save_images_png_preview,
            save_images_safetensors,
            to_0_1,
        )

        files, (start, end), formatter, preview, force, data = pwargs

        raw_file_formatter = ExportedFileName.to_formatter(
            formatter, ExportedFileName.RAW
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

        metadata = FrameTensorMetadata(-1, files, None, None)

        raw_anno_map = TensorTypes.raw_annotation_map()

        def post_frame_extration(idx: int, msg: str):
            new_frame_range.end_frame = idx
            queue.put((f"update", msg, data))

        for idx in range(start, end):
            if cancel_event.is_set():
                return

            out_file = Path(raw_file_formatter(frame_idx=idx))
            metadata.frame_idx = idx

            if not force:
                try:  # try locating prior data on disk
                    _ = load_safetensors(out_file, device, metadata, raw_anno_map)
                    msg = f"Safetensor data for frame '{idx}' already found on disk."
                    post_frame_extration(idx, msg)
                    continue  # skip this frame
                except (OSError, UserAssertionError, UserFacingError):
                    pass  # data on disk is not valid

            out_file.parent.mkdir(parents=True, exist_ok=True)

            images_list = [dec[cast(Integral, idx)].to(device) for dec in decoders]

            images_0_1: TensorTypes.ImagesTensor_0_1 = to_0_1(
                torch.stack(images_list, dim=0)
            )

            save_images_safetensors(out_file, metadata, images_0_1, None)

            if preview:
                save_images_png_preview(images_0_1, out_file)

            msg = f"Saved safetensor '{out_file}' to disk."
            post_frame_extration(idx, msg)

        for dec in decoders:
            del dec  # release resources

        data.add_frame_range(new_frame_range)

        msg = f"Frames '{start}-{end}' extracted."
        queue.put(("done", msg, None))


def process_entrypoint(*args, **kwargs):
    Mosplat_OT_extract_frame_range._operator_subprocess(*args, **kwargs)
