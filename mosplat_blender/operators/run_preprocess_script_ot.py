from __future__ import annotations

import multiprocessing as mp
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Callable, List, NamedTuple, Optional, Tuple, TypeAlias

from infrastructure.constants import PREPROCESS_SCRIPT_FUNCTION_NAME
from infrastructure.macros import (
    get_required_function,
    import_module_from_path_dynamic,
    save_tensor_stack_png_preview,
)
from infrastructure.schemas import (
    AppliedPreprocessScript,
    FrameTensorMetadata,
    ImagesTensorType,
    MediaIOMetadata,
    SavedTensorFileName,
    TensorFileFormatLookup,
    UnexpectedError,
    UserAssertionError,
    UserFacingError,
)
from operators.base_ot import MosplatOperatorBase

QueueTuple: TypeAlias = Tuple[str, str, Optional[MediaIOMetadata]]


class ProcessKwargs(NamedTuple):
    preprocess_script: Path
    media_files: List[Path]
    frame_range: Tuple[int, int]
    tensor_file_formatters: TensorFileFormatLookup
    data: MediaIOMetadata


class Mosplat_OT_run_preprocess_script(
    MosplatOperatorBase[QueueTuple, ProcessKwargs],
):
    @classmethod
    def _contexted_poll(cls, pkg) -> bool:
        props = pkg.props
        cls._poll_error_msg_list.extend(props.is_valid_media_directory_poll_result)
        cls._poll_error_msg_list.extend(props.frame_range_poll_result(pkg.prefs))
        if not pkg.props.was_frame_range_extracted:
            cls._poll_error_msg_list.append("Frame range must be extracted.")

        return len(cls._poll_error_msg_list) == 0

    def _contexted_invoke(self, pkg, event):
        prefs = pkg.prefs
        props = pkg.props

        self._media_files: List[Path] = props.media_files(prefs)
        self._preprocess_script: Path = prefs.preprocess_media_script_file_
        self._frame_range: Tuple[int, int] = props.frame_range_
        start, end = self._frame_range

        if not self.data.get_frame_range(start, end - 1):
            raise UserFacingError("Cannot run preprocess script on a function")

        self._tensor_file_formats: TensorFileFormatLookup = (
            props.generate_safetensor_filepath_formatters(
                prefs, [SavedTensorFileName.RAW, SavedTensorFileName.PREPROCESSED]
            )
        )
        return self.execute_with_package(pkg)

    def _contexted_execute(self, pkg):

        self.launch_subprocess(
            pkg.context,
            pwargs=ProcessKwargs(
                preprocess_script=self._preprocess_script,
                media_files=self._media_files,
                frame_range=self._frame_range,
                tensor_file_formatters=self._tensor_file_formats,
                data=self.data,
            ),
        )

        self.sync_to_props(pkg.props)

        return "RUNNING_MODAL"

    def _queue_callback(self, pkg, event, next):
        status, msg, new_data = next
        # sync props regardless as the updated dataclass is still valid
        if new_data:
            self.data = new_data
            self.sync_to_props(pkg.props)
        return super()._queue_callback(pkg, event, next)

    @staticmethod
    def _operator_subprocess(queue, cancel_event, *, pwargs):
        import torch
        from safetensors.torch import save_file

        script, files, (start, end), tensor_file_formats, data = pwargs
        in_file_formatter = tensor_file_formats[SavedTensorFileName.RAW]
        out_file_formatter = tensor_file_formats[SavedTensorFileName.PREPROCESSED]

        preprocess_fn = _retrieve_preprocess_fn(queue, script)
        if not preprocess_fn:
            return

        device_str: str = "cuda" if torch.cuda.is_available() else "cpu"

        # as strings and `Counter` collection type
        media_files_counter: Counter[Path] = Counter(files)

        from torchvision.utils import save_image

        for idx in range(start, end):
            if cancel_event.is_set():
                return
            try:
                in_file = Path(in_file_formatter.format(frame_idx=idx))
                out_file = Path(out_file_formatter.format(frame_idx=idx))

                tensor = _load_and_verify_tensor(
                    idx, in_file, files, media_files_counter, device_str
                )

                new_tensor: ImagesTensorType = preprocess_fn(idx, files, tensor)

                if not new_tensor.dtype == tensor.dtype:
                    raise UserAssertionError(
                        f"Data type of tensor cannot change after preprocess script",
                        expected=tensor.dtype,
                        actual=new_tensor.dtype,
                    )
                if not new_tensor.shape == tensor.shape:
                    raise UserAssertionError(
                        f"Shape of tensor cannot change after preprocess script",
                        expected=tensor.shape,
                        actual=new_tensor.shape,
                    )

                new_metadata = FrameTensorMetadata(frame_idx=idx, media_files=files)
                save_file(
                    {SavedTensorFileName._tensor_key_name(): new_tensor},
                    filename=out_file,
                    metadata=new_metadata.to_dict(),
                )

                save_tensor_stack_png_preview(new_tensor, out_file)
                queue.put(("update", f"Finished processing frame '{idx}'", None))
            except Exception as e:
                queue.put(("warning", str(e), None))
                continue

        frame_range = data.get_frame_range(start, end - 1)  # inclusive
        if not frame_range:
            msg = UnexpectedError.make_msg(f"Poll-guard failed.")
            queue.put(("error", msg, None))
        else:
            frame_range.applied_preprocess_script = (
                AppliedPreprocessScript.from_file_path(script)
            )
            queue.put(("done", f"Ran '{script}' on frames '{start}-{end}'", data))


def _load_and_verify_tensor(
    idx: int,
    in_file: Path,
    files: List[Path],
    media_files_counter: Counter[Path],
    device_str: str,
) -> ImagesTensorType:
    from safetensors import SafetensorError, safe_open

    try:
        with safe_open(in_file, framework="pt", device=device_str) as f:
            file: safe_open = f
            tensor: ImagesTensorType = file.get_tensor(
                SavedTensorFileName._tensor_key_name()
            )
            metadata: FrameTensorMetadata = FrameTensorMetadata.from_dict(
                file.metadata()
            )
    except (SafetensorError, OSError) as e:
        raise UserFacingError(
            f"Saved data in '{in_file}' is corrupted. Behavior is unpredictable. Delete the file and re-extract frame data to clean up data state.",
            e,
        ) from e

    # converts to native int
    frame_idx = metadata.frame_idx
    media_files = metadata.media_files

    if frame_idx != frame_idx:
        raise UserAssertionError(
            f"Frame index used to create '{in_file}' does not match the directory it is in.  Delete the file and re-extract frame data to clean up data state.",
            expected=idx,
            actual=frame_idx,
        )

    if media_files_counter != Counter(media_files):
        raise UserAssertionError(
            f"Media files in media directory have changed since creating '{in_file}'. Delete the file and re-extract frame data to clean up data state.",
            expected=files,
            actual=media_files,
        )

    return tensor


def _retrieve_preprocess_fn(
    queue: mp.Queue[QueueTuple], script: Path
) -> Optional[Callable]:
    try:
        module: ModuleType = import_module_from_path_dynamic(script)
        preprocess_fn: Callable = get_required_function(
            module, PREPROCESS_SCRIPT_FUNCTION_NAME
        )
        queue.put(
            (
                "update",
                f"Found and imported function '{preprocess_fn.__qualname__}' from module '{module.__name__}'.",
                None,
            )
        )
    except ImportError as e:
        msg = UserFacingError.make_msg(
            f"Cannot import selected preprocess script: '${script}'",
            e,
        )
        queue.put(("error", msg, None))
        return None
    except (AttributeError, TypeError) as e:
        msg = UserFacingError.make_msg(
            f"Cannot import required `${PREPROCESS_SCRIPT_FUNCTION_NAME}` function from selected preprocess script: '{script}'",
            e,
        )
        queue.put(("error", msg, None))
        return None

    return preprocess_fn


def process_entrypoint(*args, **kwargs):
    Mosplat_OT_run_preprocess_script._operator_subprocess(*args, **kwargs)
