"""functions here should raise standard library error types."""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_file_location
from os import stat_result
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Callable,
    Counter,
    Iterable,
    List,
    LiteralString,
    Optional,
    Set,
    Tuple,
    TypeAlias,
    TypeGuard,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from infrastructure.schemas import ImagesTensorType

T = TypeVar("T")
K = TypeVar("K", bound=Tuple)


def int_median(iter: Iterable[int]) -> int:
    return int(round(median(iter))) if iter else -1


L = TypeVar("L", bound=LiteralString)
I = TypeVar("I")

Immutable: TypeAlias = Union[Tuple[I, ...], I]

# set is effectively immutable-like for our purposes
ImmutableLike: TypeAlias = Union[Immutable[L], Set[L]]


def immutable_to_set(im: ImmutableLike[L]) -> Set[L]:
    if isinstance(im, set):
        return im

    return set(im) if isinstance(im, tuple) else {im}


def append_if_not_equals(iter: List[T], *, item: T, target: T) -> None:
    if item != target:
        iter.append(item)


def try_access_path(p: Path) -> stat_result:
    """
    will raise `FileNotFoundError`, `OSError`, or `PermissionError`.
    returns the stat result of the file IF it is accessible.
    """

    if not p.exists():
        raise FileNotFoundError(f"'{p}' does not exist.")

    try:
        return p.stat()  # will raise either `OSError` or `PermissionError`
    except (OSError, PermissionError) as e:
        e.add_note(f"NOTE: '{e}' was found but could not retrieve a stat result.")
        raise


def is_path_accessible(p: Path) -> bool:
    try:
        try_access_path(p)
        return True
    except (OSError, PermissionError, FileNotFoundError):
        return False


def tuple_type_matches_known_tuple_type(
    unknown_tuple: Tuple, known_tuple: K
) -> TypeGuard[K]:
    if len(unknown_tuple) != len(known_tuple):
        return False
    return all(type(v) == type(t) for v, t in zip(unknown_tuple, known_tuple))


def kill_subprocess_cross_platform(pid: int):
    if sys.platform != "win32":
        os.killpg(pid, signal.SIGKILL)  # TODO: cross-platform check
    else:
        import psutil

        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except psutil.NoSuchProcess:
            pass  # this requires no exception-raising


def import_module_from_path_dynamic(path: Path) -> ModuleType:
    """raises `ImportError` if a specification could not be loaded from the given path"""
    path = path.resolve()

    spec: Optional[ModuleSpec] = spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from '{path}'")

    module: ModuleType = module_from_spec(spec)
    sys.modules[path.stem] = module  # add to current sys modules
    spec.loader.exec_module(module)  # get the

    return module


def get_required_function(module: ModuleType, name: str) -> Callable:
    try:
        fn = getattr(module, name)
    except AttributeError as e:
        e.add_note(f"NOTE: Module '{module.__name__}' has no function '{name}'")
        raise

    if not isinstance(fn, Callable):
        raise TypeError(f"'{name}' exists but is not callable")

    return fn


def save_tensor_stack_png_preview(tensor: ImagesTensorType, tensor_out_file: Path):
    from torchvision.utils import save_image

    preview_png_file: Path = tensor_out_file.parent / f"{tensor_out_file.stem}.png"

    save_image(tensor, preview_png_file)


def load_and_verify_tensor(
    idx: int,
    in_file: Path,
    files: List[Path],
    media_files_counter: Counter[Path],
    device_str: str,
) -> ImagesTensorType:
    from safetensors import SafetensorError, safe_open

    from infrastructure.schemas import (
        FrameTensorMetadata,
        SavedTensorFileName,
        UserAssertionError,
        UserFacingError,
    )

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


def pad_tensor(images: ImagesTensorType, multiple: int) -> ImagesTensorType:
    import torch

    _, _, H, W = images.shape
    pad_height = (multiple - H % multiple) % multiple
    pad_width = (multiple - W % multiple) % multiple

    return torch.nn.functional.pad(
        images, (0, pad_width, 0, pad_height), mode="constant", value=0.0
    )


def crop_tensor(
    images: ImagesTensorType,
    *,
    max_size: int,
    multiple: int,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> ImagesTensorType:
    import torch.nn.functional as F

    _, _, H, W = images.shape

    # crop to max size
    max_hw = max(H, W)
    if max_hw > max_size:
        scale = max_size / max_hw
        new_H = int(round(H * scale))
        new_W = int(round(W * scale))

        images = F.interpolate(
            images,
            size=(new_H, new_W),
            mode=mode,
            align_corners=align_corners if mode in ("bilinear", "bicubic") else None,
        )
        H, W = new_H, new_W

    crop_H = (H // multiple) * multiple
    crop_W = (W // multiple) * multiple

    # pad to 14
    top = (H - crop_H) // 2
    left = (W - crop_W) // 2

    images = images[:, :, top : top + crop_H, left : left + crop_W]

    return images
