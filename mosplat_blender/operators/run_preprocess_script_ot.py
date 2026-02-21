from __future__ import annotations

import multiprocessing as mp
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import (
    Callable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeAlias,
    assert_never,
)

from ..infrastructure.constants import PREPROCESS_SCRIPT_FUNCTION_NAME
from ..infrastructure.macros import (
    get_required_function,
    import_module_from_path_dynamic,
    load_and_verify_tensor_file,
    save_images_tensor,
    save_tensor_stack_png_preview,
    to_0_1,
)
from ..infrastructure.schemas import (
    AppliedPreprocessScript,
    FrameTensorMetadata,
    ImagesMaskTensor,
    ImagesTensor_0_1,
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

        device_str: str = "cuda" if torch.cuda.is_available() else "cpu"

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
                        device_str,
                        new_metadata,
                        keys=[
                            SavedTensorKey.IMAGES,
                            SavedTensorKey.IMAGES_MASK,
                        ],
                    )
                    queue.put(
                        (
                            "update",
                            f"Previous preprocessed data found on disk for frame '{idx}'",
                            None,
                        )
                    )
                    continue
                except (OSError, UserAssertionError, UserFacingError):
                    pass

                validation_metadata: FrameTensorMetadata = FrameTensorMetadata(
                    idx, files, preprocess_script=None, model_options=None
                )  # preprocess script did not exist in extraction step

                tensors = load_and_verify_tensor_file(
                    in_file,
                    device_str,
                    validation_metadata,
                    keys=[SavedTensorKey.IMAGES],
                )
                images_0_255 = tensors[SavedTensorKey.IMAGES]
                images_0_1: ImagesTensor_0_1 = to_0_1(images_0_255)

                output = preprocess_fn(idx, files, images_0_1)

                images, images_mask = _validate_preprocess_script_output(
                    output, images_0_1
                )
                save_images_tensor(out_file, new_metadata, images, images_mask)

                if preview:
                    save_tensor_stack_png_preview(images, out_file)
                    save_tensor_stack_png_preview(
                        images * images_mask, out_file, ".masked"
                    )
                queue.put(("update", f"Finished processing frame '{idx}'", None))
            except Exception as e:
                msg = UserFacingError.make_msg(
                    f"Error ocurred while running preprocess script on frame '{idx}'.",
                    e,
                )
                queue.put(("warning", msg, None))
                continue

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


def _validate_preprocess_script_output(
    returned, input_images: ImagesTensor_0_1
) -> Tuple[ImagesTensor_0_1, ImagesMaskTensor]:
    import torch

    if not isinstance(returned, tuple):
        raise UserAssertionError(
            f"Return value of preprocess script must be a tuple",
            expected=tuple.__name__,
            actual=type(returned).__name__,
        )
    if not len(returned) == 2:
        raise UserAssertionError(
            f"Return value of preprocess script must be a tuple of size 2",
            expected=len(returned),
            actual=2,
        )

    returned_images, returned_mask = returned

    # validate returned images
    if not isinstance(returned_images, torch.Tensor):
        raise UserAssertionError(
            f"Returned images of preprocess script must be a torch tensor",
            expected=torch.Tensor.__name__,
            actual=type(returned_images).__name__,
        )
    if not returned_images.dtype == input_images.dtype:
        raise UserAssertionError(
            f"Data type of images tensor cannot change after preprocess script",
            expected=input_images.dtype,
            actual=returned_images.dtype,
        )
    if not returned_images.shape == input_images.shape:
        raise UserAssertionError(
            f"Shape of images tensor cannot change after preprocess script",
            expected=returned_images.shape,
            actual=returned_images.shape,
        )

    if returned_mask is None:
        returned_mask = torch.ones_like(returned_images[:, :1], dtype=torch.bool)
    if not returned_mask.dtype == torch.bool:
        raise UserAssertionError(
            f"Data type of images mask tensor incorrect",
            expected=torch.bool,
            actual=returned_images.dtype,
        )
    B, _, H, W = input_images.shape
    expected_shape = torch.Size((B, 1, H, W))
    if not returned_mask.shape == expected_shape:
        raise UserAssertionError(
            f"Shape of images mask tensor incorrect",
            expected=expected_shape,
            actual=returned_images.shape,
        )

    return returned_images, returned_mask


def process_entrypoint(*args, **kwargs):
    Mosplat_OT_run_preprocess_script._operator_subprocess(*args, **kwargs)
