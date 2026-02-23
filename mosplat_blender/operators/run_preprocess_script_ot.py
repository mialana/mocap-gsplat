from __future__ import annotations

import multiprocessing as mp
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Callable, List, NamedTuple, Optional, Tuple, TypeAlias

from ..infrastructure.constants import PREPROCESS_SCRIPT_FUNCTION_NAME
from ..infrastructure.macros import (
    get_required_function,
    import_module_from_path_dynamic,
)
from ..infrastructure.schemas import (
    AppliedPreprocessScript,
    FrameTensorMetadata,
    MediaIOMetadata,
    SavedTensorFileName,
    SavedTensorKey,
    UnexpectedError,
    UserAssertionError,
    UserFacingError,
)
from .base_ot import MosplatOperatorBase

QueueTuple: TypeAlias = Tuple[str, str, Optional[MediaIOMetadata]]


class ProcessKwargs(NamedTuple):
    preprocess_script: Path
    media_files: List[Path]
    frame_range: Tuple[int, int]
    exported_file_formatter: str
    create_preview_images: bool
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

        self._exported_file_formattter: str = props.exported_file_formatter(prefs)
        return self.execute_with_package(pkg)

    def _contexted_execute(self, pkg):

        self.launch_subprocess(
            pkg.context,
            pwargs=ProcessKwargs(
                preprocess_script=self._preprocess_script,
                media_files=self._media_files,
                frame_range=self._frame_range,
                exported_file_formatter=self._exported_file_formattter,
                create_preview_images=bool(pkg.prefs.create_preview_images),
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

        from ..infrastructure.dl_ops import (
            TensorTypes as TensorTypes,
            load_and_verify_tensor_file,
            save_images_png_preview_stacked,
            save_images_tensor,
            to_0_1,
            validate_preprocess_script_output,
        )

        script, files, (start, end), exported_file_formatter, preview, data = pwargs

        in_file_formatter = partial(
            exported_file_formatter.format,
            file_name=SavedTensorFileName.RAW,
            file_ext="safetensors",
        )
        out_file_formatter = partial(
            exported_file_formatter.format,
            file_name=SavedTensorFileName.PREPROCESSED,
            file_ext="safetensors",
        )

        preprocess_fn = _retrieve_preprocess_fn(queue, script)
        if not preprocess_fn:
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        applied_preprocess_script = AppliedPreprocessScript.from_file_path(script)

        for idx in range(start, end):
            if cancel_event.is_set():
                return
            try:
                in_file = Path(in_file_formatter(frame_idx=idx))
                out_file = Path(out_file_formatter(frame_idx=idx))

                new_metadata: FrameTensorMetadata = FrameTensorMetadata(
                    idx,
                    files,
                    preprocess_script=applied_preprocess_script,
                    model_options=None,
                )

                try:
                    _ = load_and_verify_tensor_file(
                        out_file,
                        device,
                        new_metadata,
                        map={
                            SavedTensorKey.IMAGES.value: TensorTypes.annotation_of(
                                TensorTypes.ImagesTensor_0_255
                            ),
                            SavedTensorKey.IMAGES_ALPHA.value: TensorTypes.annotation_of(
                                TensorTypes.ImagesAlphaTensor_0_255
                            ),
                        },
                    )
                    queue.put(
                        (
                            "update",
                            f"Previous preprocessed data found on disk for frame '{idx}'",
                            None,
                        )
                    )
                except (OSError, UserAssertionError, UserFacingError):
                    validation_metadata: FrameTensorMetadata = FrameTensorMetadata(
                        idx, files, preprocess_script=None, model_options=None
                    )  # preprocess script did not exist in extraction step

                    tensors = load_and_verify_tensor_file(
                        in_file,
                        device,
                        validation_metadata,
                        map={
                            SavedTensorKey.IMAGES.value: TensorTypes.annotation_of(
                                TensorTypes.ImagesTensor_0_255
                            )
                        },
                    )
                    raw_images_0_1: TensorTypes.ImagesTensor_0_1 = to_0_1(
                        tensors[SavedTensorKey.IMAGES]
                    )

                    output = preprocess_fn(idx, files, raw_images_0_1)

                    images_0_1, images_alpha_0_1 = validate_preprocess_script_output(
                        output, raw_images_0_1
                    )

                    save_images_tensor(
                        out_file, new_metadata, images_0_1, images_alpha_0_1
                    )

                    if preview:
                        save_images_png_preview_stacked(images_0_1, out_file)
                        save_images_png_preview_stacked(
                            images_0_1 * images_alpha_0_1, out_file, ".masked"
                        )
                    queue.put(("update", f"Finished processing frame '{idx}'", None))
            except Exception as e:
                msg = UserFacingError.make_msg(
                    f"Error ocurred while running preprocess script on frame '{idx}'.",
                    e,
                )
                queue.put(("warning", msg, None))

        frame_range = data.query_frame_range(start, end - 1)  # inclusive
        if not frame_range or len(frame_range) > 1:
            msg = UnexpectedError.make_msg("Poll-guard failed.")
            queue.put(("error", msg, None))
            return
        else:
            frame_range[0].applied_preprocess_script = applied_preprocess_script
            queue.put(("done", f"Ran '{script}' on frames '{start}-{end}'", data))


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
