from pathlib import Path
from typing import Counter, List, NamedTuple, Tuple

from infrastructure.macros import load_and_verify_default_tensor, load_and_verify_tensor
from infrastructure.schemas import (
    AppliedPreprocessScript,
    FrameTensorMetadata,
    PointCloudTensors,
    SavedTensorFileName,
    TensorFileFormatLookup,
    UserAssertionError,
    UserFacingError,
    VGGTModelOptions,
)
from interfaces import VGGTInterface
from operators.base_ot import MosplatOperatorBase


class ThreadKwargs(NamedTuple):
    preprocess_script: Path
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
        self._preprocess_script: Path = prefs.preprocess_media_script_file_

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
                preprocess_script=self._preprocess_script,
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
        from safetensors.torch import save_file

        script, files, (start, end), tensor_file_formats, options = twargs
        in_file_formatter = tensor_file_formats[SavedTensorFileName.PREPROCESSED]
        out_file_formatter = tensor_file_formats[SavedTensorFileName.MODEL_INFERENCE]

        device_str: str = "cuda" if torch.cuda.is_available() else "cpu"

        applied_preprocess_script = AppliedPreprocessScript.from_file_path(script)

        for idx in range(start, end):
            if cancel_event.is_set():
                return
            try:
                in_file = Path(in_file_formatter.format(frame_idx=idx))
                out_file = Path(out_file_formatter.format(frame_idx=idx))
                new_metadata: FrameTensorMetadata = FrameTensorMetadata(
                    idx, files, applied_preprocess_script, options
                )

                try:
                    _ = load_and_verify_tensor(out_file, device_str, new_metadata)
                    queue.put(
                        (
                            "update",
                            f"Previous point cloud inference data found on disk for frame '{idx}'",
                        )
                    )
                    continue
                except (OSError, UserAssertionError):
                    pass

                validation_metadata: FrameTensorMetadata = FrameTensorMetadata(
                    idx,
                    files,
                    preprocess_script=applied_preprocess_script,
                    model_options=None,
                )  # options did not exist in 'run preprocess script' step
                images_tensor = load_and_verify_default_tensor(
                    in_file, device_str, validation_metadata
                )
                if images_tensor is None:
                    raise RuntimeError("Poll-guard failed.")

                pc_tensors: PointCloudTensors = VGGTInterface().run_inference(
                    images_tensor, new_metadata, options
                )
                save_file(
                    pc_tensors.to_dict(),
                    out_file,
                    metadata=new_metadata.to_dict(),
                )

                queue.put(("update", f"Ran inference on frame '{idx}'"))
            except Exception as e:
                msg = UserFacingError.make_msg(
                    f"Error ocurred while running inference on frame '{idx}'.", e
                )
                queue.put(("warning", msg))
                continue

        queue.put(("done", "Inference complete for current frame range."))


def process_entrypoint(*args, **kwargs):
    Mosplat_OT_run_inference._operator_subprocess(*args, **kwargs)
