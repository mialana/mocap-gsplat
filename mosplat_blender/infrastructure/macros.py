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
    Dict,
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
    import torch
    from jaxtyping import Float32, UInt8
    from safetensors import safe_open

    from infrastructure.schemas import FrameTensorMetadata, ImagesTensorType

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


def save_tensor_stack_separate_png_previews(
    tensor: ImagesTensorType, tensor_out_file: Path
):
    from torchvision.utils import save_image

    for idx, img in enumerate(tensor):
        preview_png_file: Path = (
            tensor_out_file.parent / f"{tensor_out_file.stem}.{idx:04d}.png"
        )
        save_image(img, preview_png_file)


def load_and_verify_tensor(
    in_file: Path, device_str: str, new_metadata: FrameTensorMetadata
) -> Dict[str, torch.Tensor]:
    from safetensors import SafetensorError, safe_open

    from infrastructure.schemas import (
        FrameTensorMetadata,
        UserAssertionError,
        UserFacingError,
    )

    try:
        try_access_path(in_file)
    except (FileNotFoundError, OSError, PermissionError) as e:
        e.add_note("Restart from extraction process if necessary.")
        raise OSError from e

    try:
        tensors: Dict[str, torch.Tensor] = {}
        with safe_open(in_file, framework="pt", device=device_str) as f:
            file: safe_open = f
            for key in file.keys():
                tensors |= {key: file.get_tensor(key)}

            saved_metadata: FrameTensorMetadata = FrameTensorMetadata.from_dict(
                file.metadata()
            )
    except (SafetensorError, OSError) as e:
        raise UserFacingError(
            f"Saved data in '{in_file}' is corrupted. Behavior is unpredictable. Delete the file and re-extract frame data to clean up data state.",
            e,
        ) from e

    for idx, item in enumerate(new_metadata):
        saved_item = saved_metadata[idx]
        if item != saved_item:
            raise UserAssertionError(
                f"Metadata used to create '{in_file}' does not match the new desired metadata. To clean up data state, delete the file and re-extract frame data",
                expected=idx,
                actual=saved_metadata.frame_idx,
            )

    return tensors


def load_and_verify_default_tensor(
    in_file: Path, device_str: str, new_metadata: FrameTensorMetadata
) -> Optional[ImagesTensorType]:
    from infrastructure.schemas import SavedTensorFileName

    tensors = load_and_verify_tensor(in_file, device_str, new_metadata)
    return tensors.get(SavedTensorFileName._default_tensor_key())


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


def save_ply_ascii(
    out_file: Path, xyz: Float32[torch.Tensor, "N 3"], rgb: UInt8[torch.Tensor, "N 3"]
):
    assert xyz.shape[0] == rgb.shape[0], "'xyz' tensor shape dim 0 should match 'rgb'."
    assert (
        xyz.shape[1] == 3 and rgb.shape[1] == 3
    ), "'xyz' and 'rgb' should have shape 3 in last dimension."

    xyz = xyz.cpu()
    rgb = rgb.cpu()

    N = xyz.shape[0]

    with open(out_file, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(N):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


def save_ply_binary(
    out_file: Path, xyz: Float32[torch.Tensor, "N 3"], rgb: UInt8[torch.Tensor, "N 3"]
):
    import numpy as np

    assert xyz.dtype == torch.float32
    assert rgb.dtype == torch.uint8
    assert xyz.shape == rgb.shape
    assert xyz.device.type == "cpu"

    N = xyz.shape[0]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    with open(out_file, "wb") as f:
        f.write(header.encode("ascii"))

        xyz_bytes = xyz.view(torch.uint8)
        rgb_bytes = rgb

        for i in range(N):
            f.write(xyz_bytes[i].cpu().numpy().astype(np.float32).tobytes())
            f.write(rgb_bytes[i].cpu().numpy().astype(np.uint8).tobytes())
