"""functions here should raise standard library error types."""

from __future__ import annotations

import os
import signal
import sys
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Callable,
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
    from jaxtyping import Float32, UInt8
    from torch import Device, Tensor

    # type-check only imports to avoid circular imports
    from .schemas import CropGeometry, FrameTensorMetadata, TensorTypes as TT


T = TypeVar("T")
K = TypeVar("K", bound=Tuple)

L = TypeVar("L", bound=LiteralString)
I = TypeVar("I")

Immutable: TypeAlias = Union[Tuple[I, ...], I]

# set is effectively immutable-like for our purposes
ImmutableLike: TypeAlias = Union[Immutable[L], Set[L]]


def int_median(iter: Iterable[int]) -> int:
    return int(round(median(iter))) if iter else -1


def immutable_to_set(im: ImmutableLike[L]) -> Set[L]:
    if isinstance(im, set):
        return im

    return set(im) if isinstance(im, tuple) else {im}


def append_if_not_equals(iter: List[T], *, item: T, target: T) -> None:
    if item != target:
        iter.append(item)


def try_access_path(p: Path) -> os.stat_result:
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


def to_0_1(tensor: Tensor) -> Tensor:
    import torch

    if tensor.dtype == torch.float32:
        return tensor

    assert (
        tensor.dtype == torch.uint8
    ), "Given tensor should only ever be of type float32 or uint8"

    return tensor.to(torch.float32).div_(255.0)


def to_0_255(tensor: Tensor) -> Tensor:
    import torch

    if tensor.dtype == torch.uint8:
        return tensor

    assert (
        tensor.dtype == torch.float32
    ), "Given tensor should only ever be of type float32 or uint8"

    return (tensor.clamp(0, 1).mul_(255.0)).round().to(torch.uint8)


def to_channel_as_primary(tensor: Tensor) -> Tensor:
    assert len(tensor.shape) == 4
    return tensor.permute(0, 3, 1, 2)


def to_channel_as_item(tensor: Tensor) -> Tensor:
    assert len(tensor.shape) == 4
    return tensor.permute(0, 2, 3, 1)


def save_tensor_stack_png_preview(
    tensor: TT.ImagesTensor_0_1, tensor_out_file: Path, suffix: str = ""
):
    from torchvision.utils import save_image

    tensor_0_1 = to_0_1(tensor)

    preview_png_file: Path = (
        tensor_out_file.parent / f"{tensor_out_file.stem}{suffix}.png"
    )

    save_image(tensor_0_1, preview_png_file, nrow=4)


def save_images_tensor(
    out_file: Path,
    metadata: FrameTensorMetadata,
    images_0_1: TT.ImagesAlphaTensor_0_1,
    images_alpha_0_1: Optional[TT.ImagesAlphaTensor_0_1],
):
    from safetensors.torch import save_file

    from .schemas import SavedTensorKey

    images = to_0_255(images_0_1)
    out_tensors = {SavedTensorKey.IMAGES.value: images}

    if images_alpha_0_1 is not None:
        images_alpha = to_0_255(images_alpha_0_1)
        out_tensors |= {SavedTensorKey.IMAGES_ALPHA.value: images_alpha}

    save_file(
        out_tensors,
        filename=out_file,
        metadata=metadata.to_dict(),
    )


def load_and_verify_tensor_file(
    in_file: Path,
    device: Device,
    new_metadata: FrameTensorMetadata,
    keys: List[str],  # keys to retrieve from the file
) -> Dict[str, Tensor]:
    from safetensors import SafetensorError, safe_open

    from .schemas import (
        FrameTensorMetadata,
        UserAssertionError,
        UserFacingError,
    )

    try:
        try_access_path(in_file)
    except (FileNotFoundError, OSError, PermissionError) as e:
        e.add_note("Restart from extraction process if necessary.")
        raise OSError from e

    tensors: Dict[str, Tensor] = {}
    with safe_open(in_file, framework="pt", device=str(device)) as f:
        file: safe_open = f
        for key in keys:
            try:
                tensors |= {key: file.get_tensor(key)}
            except SafetensorError as e:
                raise UserFacingError from e

        try:
            saved_metadata: FrameTensorMetadata = FrameTensorMetadata.from_dict(
                file.metadata()
            )
        except TypeError as e:
            raise UserFacingError(
                f"Metadata used to create '{in_file}' does not match the structure of new desired metadata.",
                e,
            ) from e

    for idx, item in enumerate(new_metadata):
        saved_item = saved_metadata[idx]
        if item != saved_item:
            raise UserAssertionError(
                f"A field used to create metadata '{in_file}' does not match the corresponding field in new desired metadata.",
                expected=item,
                actual=saved_item,
            )

    return tensors


def crop_tensor(
    tensor: Tensor,  # limited by torch `interpolate`, i.e. floating-point data types
    crop_geom: CropGeometry,
    *,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> Tensor:
    import torch.nn.functional as F

    assert len(tensor.shape) == 4
    _, _, H, W = tensor.shape
    assert H == crop_geom.orig_H and W == crop_geom.orig_W

    # crop to max size
    max_dim: int = max(crop_geom.orig_H, crop_geom.orig_W)
    if max_dim > crop_geom.max_size:
        tensor = F.interpolate(
            tensor,
            size=(crop_geom.new_H, crop_geom.new_W),
            mode=mode,
            align_corners=align_corners if mode in ("bilinear", "bicubic") else None,
        )

    tensor = tensor[
        :,
        :,
        crop_geom.top : crop_geom.top + crop_geom.crop_H,
        crop_geom.left : crop_geom.left + crop_geom.crop_W,
    ]

    return tensor


def save_ply_ascii(out_file: Path, xyz: TT.XYZTensor, rgb_0_255: TT.RGBTensor):
    assert xyz.shape[0] == rgb_0_255.shape[0]
    assert xyz.shape[1] == 3 and rgb_0_255.shape[1] == 3

    xyz = xyz.cpu()
    rgb_0_255 = rgb_0_255.cpu()

    N = xyz.shape[0]

    with out_file.open(mode="w") as f:
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
            r, g, b = rgb_0_255[i]
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


def save_ply_binary(
    out_file: Path, xyz: Float32[Tensor, "N 3"], rgb: UInt8[Tensor, "N 3"]
):
    import numpy as np
    import torch

    rgb_0_255 = to_0_255(rgb)
    assert xyz.dtype == torch.float32
    assert xyz.shape == rgb.shape

    xyz_np = xyz.cpu().numpy()
    rgb_np = rgb_0_255.cpu().numpy()

    N = xyz_np.shape[0]

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

    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )

    with out_file.open(mode="wb") as f:
        f.write(header.encode("ascii"))

        vertices = np.empty(N, dtype=vertex_dtype)
        vertices["x"] = xyz_np[:, 0]
        vertices["y"] = xyz_np[:, 1]
        vertices["z"] = xyz_np[:, 2]
        vertices["red"] = rgb_np[:, 0]
        vertices["green"] = rgb_np[:, 1]
        vertices["blue"] = rgb_np[:, 2]

        f.write(vertices.tobytes())
