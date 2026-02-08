from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Callable, List, NamedTuple, Optional, Tuple

from infrastructure.constants import PREPROCESS_SCRIPT_FUNCTION_NAME
from infrastructure.macros import (
    get_required_function,
    import_module_from_path_dynamic,
)
from infrastructure.schemas import (
    DeveloperError,
    FrameTensorMetadata,
    MediaIOMetadata,
    SavedTensorFileName,
    TensorFileFormatLookup,
    UserFacingError,
)
from operators.base_ot import MosplatOperatorBase


class ProcessKwargs(NamedTuple):
    preprocess_script: Path
    media_files: List[Path]
    frame_range: Tuple[int, int]
    tensor_file_formatters: TensorFileFormatLookup
    data: MediaIOMetadata


class Mosplat_OT_run_preprocess_script(
    MosplatOperatorBase[Tuple[str, str, Optional[MediaIOMetadata]], ProcessKwargs],
):
    @classmethod
    def _contexted_poll(cls, pkg) -> bool:
        props = pkg.props
        cls._poll_error_msg_list.extend(props.is_valid_media_directory_poll_result)
        cls._poll_error_msg_list.extend(props.frame_range_poll_result(pkg.prefs))

        return len(cls._poll_error_msg_list) == 0

    def _contexted_invoke(self, pkg, event):
        prefs = pkg.prefs
        props = pkg.props

        self._media_files: List[Path] = props.media_files(prefs)
        self._preprocess_script: Path = prefs.preprocess_media_script_file_
        self._frame_range: Tuple[int, int] = tuple(props.frame_range)
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
        from safetensors import SafetensorError, safe_open
        from safetensors.torch import save_file

        script, files, (start, end), tensor_file_formats, data = pwargs

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
            return
        except (AttributeError, TypeError) as e:
            msg = UserFacingError.make_msg(
                f"Cannot import required `${PREPROCESS_SCRIPT_FUNCTION_NAME}` function from selected preprocess script: '{script}'",
                e,
            )
            queue.put(("error", msg, None))
            return

        in_file_formatter = tensor_file_formats[SavedTensorFileName.RAW]
        out_file_formatter = tensor_file_formats[SavedTensorFileName.PREPROCESSED]

        device_str: str = "cuda" if torch.cuda.is_available() else "cpu"

        # as strings and `Counter` collection type
        media_files_counter = Counter(pwargs.media_files)

        for idx in range(start, end):
            if cancel_event.is_set():
                return
            try:
                in_file = in_file_formatter.format(frame_idx=idx)
                out_file = out_file_formatter.format(frame_idx=idx)
                try:
                    with safe_open(in_file, framework="pt", device=device_str) as f:
                        file: safe_open = f
                        tensor: torch.Tensor = file.get_tensor(
                            SavedTensorFileName._tensor_key_name()
                        )
                        metadata: FrameTensorMetadata = FrameTensorMetadata.from_dict(
                            file.metadata()
                        )
                except (SafetensorError, OSError) as e:
                    raise UserFacingError(
                        f"Saved data in '{in_file_formatter}' is corrupted. Behavior is unpredictable. Delete the file and re-extract frame data to clean up data state.",
                        e,
                    ) from e

                # converts to native int
                frame_idx = metadata.frame_idx
                media_files = metadata.media_files

                if frame_idx != idx:
                    raise ValueError(
                        f"Frame index used to create '{in_file_formatter}' does not match the directory it is in.  Delete the file and re-extract frame data to clean up data state.",
                        f"\nExpected Index: '{idx}'"
                        f"\nExtracted Index: '{frame_idx}'",
                    )

                if media_files_counter != Counter(media_files):
                    raise ValueError(
                        f"Media files in media directory have changed since creating '{in_file_formatter}'. Delete the file and re-extract frame data to clean up data state.",
                        f"\nExpected Files: '{pwargs.media_files}'"
                        f"\nMetadata Files: '{media_files}'",
                    )

                new_tensor = preprocess_fn(frame_idx, media_files, tensor)
                new_metadata = FrameTensorMetadata(frame_idx=idx, media_files=files)
                save_file(
                    {SavedTensorFileName._tensor_key_name(): new_tensor},
                    filename=out_file,
                    metadata=new_metadata.to_dict(),
                )
                queue.put(("update", f"Finished processing frame '{frame_idx}'", None))
            except Exception as e:
                msg = UserFacingError.make_msg(
                    f"Could not run script on frame '{idx}'", e
                )
                queue.put(("error", msg, None))
                continue

        frame_range = data.get_frame_range(start, end)
        if not frame_range:
            msg = DeveloperError.make_msg(
                f"Should not have ran script on non-existing frame range."
            )
            queue.put(("error", msg, None))
        else:
            queue.put(("done", f"Ran '{script}' on frames '{start}-{end}'", data))


def process_entrypoint(*args, **kwargs):
    Mosplat_OT_run_preprocess_script._operator_subprocess(*args, **kwargs)
