from pathlib import Path
from typing import Counter, List, NamedTuple, Tuple

from infrastructure.macros import load_and_verify_tensor
from infrastructure.schemas import (
    FrameTensorMetadata,
    SavedTensorFileName,
    TensorFileFormatLookup,
    VGGTModelOptions,
)
from interfaces import VGGTInterface
from operators.base_ot import MosplatOperatorBase


class ThreadKwargs(NamedTuple):
    media_files: List[Path]
    frame_range: Tuple[int, int]
    tensor_file_formatters: TensorFileFormatLookup
    model_options: VGGTModelOptions


class Mosplat_OT_run_inference(MosplatOperatorBase[Tuple[str, str], ThreadKwargs]):
    @classmethod
    def _contexted_poll(cls, pkg):
        if VGGTInterface().model is None:
            cls._poll_error_msg_list.append("Model must be initialized.")
        if not pkg.props.was_frame_range_extracted:
            cls._poll_error_msg_list.append("Frame range must be extracted.")

        return len(cls._poll_error_msg_list) == 0

    def _queue_callback(self, pkg, event, next):

        return super()._queue_callback(pkg, event, next)

    def _contexted_invoke(self, pkg, event):
        prefs = pkg.prefs
        props = pkg.props

        self._media_files: List[Path] = props.media_files(prefs)
        self._frame_range: Tuple[int, int] = props.frame_range_

        self._tensor_file_formats: TensorFileFormatLookup = (
            props.generate_safetensor_filepath_formatters(
                prefs,
                [SavedTensorFileName.PREPROCESSED, SavedTensorFileName.MODEL_INFERENCE],
            )
        )
        return self.execute_with_package(pkg)

    def _contexted_execute(self, pkg):
        self.launch_thread(
            pkg.context,
            twargs=ThreadKwargs(
                media_files=self._media_files,
                frame_range=self._frame_range,
                tensor_file_formatters=self._tensor_file_formats,
                model_options=pkg.props.options_accessor.to_dataclass(),
            ),
        )

        return "RUNNING_MODAL"

    @staticmethod
    def _operator_thread(queue, cancel_event, *, twargs):
        import torch

        files, (start, end), tensor_file_formats, options = twargs
        in_file_formatter = tensor_file_formats[SavedTensorFileName.PREPROCESSED]
        out_file_formatter = tensor_file_formats[SavedTensorFileName.MODEL_INFERENCE]

        device_str: str = "cuda" if torch.cuda.is_available() else "cpu"

        # as strings and `Counter` collection type
        media_files_counter: Counter[Path] = Counter(files)

        for idx in range(start, end):
            if cancel_event.is_set():
                return
            try:
                in_file = Path(in_file_formatter.format(frame_idx=idx))
                out_file = Path(out_file_formatter.format(frame_idx=idx))

                images_tensor = load_and_verify_tensor(
                    idx, in_file, files, media_files_counter, device_str
                )
                metadata: FrameTensorMetadata = FrameTensorMetadata(idx, files)

                pc_tensors = VGGTInterface().run_inference(
                    images_tensor, metadata, options
                )

                queue.put(("update", f"Ran inference on frame '{idx}'"))
            except Exception as e:
                queue.put(("warning", str(e)))
                continue


def process_entrypoint(*args, **kwargs):
    Mosplat_OT_run_inference._operator_subprocess(*args, **kwargs)
