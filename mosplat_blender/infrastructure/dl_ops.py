"""deep-learning operations utilizing PyTorch tensors. should use nested import"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import (
    Annotated,
    Dict,
    Optional,
    Self,
    TypeAlias,
    TypeVar,
    get_args,
    get_type_hints,
)

import dltype
import torch
from dltype._lib._core import DLTypeAnnotation

from .macros import try_access_path
from .schemas import CropGeometry, FrameTensorMetadata


class TensorTypes:
    ImagesTensor_0_1: TypeAlias = Annotated[
        torch.Tensor, dltype.Float32Tensor["S 3 H W"]
    ]
    ImagesTensor_0_255: TypeAlias = Annotated[
        torch.Tensor, dltype.UInt8Tensor["S 3 H W"]
    ]

    ImagesAlphaTensor_0_1: TypeAlias = Annotated[
        torch.Tensor, dltype.Float32Tensor["S 1 H W"]
    ]
    ImagesAlphaTensor_0_255: TypeAlias = Annotated[
        torch.Tensor, dltype.UInt8Tensor["S 1 H W"]
    ]

    VoxelTensor: TypeAlias = Annotated[torch.Tensor, dltype.Float32Tensor[None]]

    XYZTensor: TypeAlias = Annotated[torch.Tensor, dltype.Float32Tensor["N 3"]]
    RGB_0_255_Tensor: TypeAlias = Annotated[torch.Tensor, dltype.UInt8Tensor["N 3"]]
    ConfTensor: TypeAlias = Annotated[torch.Tensor, dltype.Float32Tensor["N"]]
    PointCamsTensor: TypeAlias = Annotated[torch.Tensor, dltype.Int32Tensor["N"]]

    ExtrinsicTensor: TypeAlias = Annotated[torch.Tensor, dltype.Float32Tensor["S 3 4"]]
    IntrinsicTensor: TypeAlias = Annotated[torch.Tensor, dltype.Float32Tensor["S 3 3"]]

    DepthTensor: TypeAlias = Annotated[torch.Tensor, dltype.Float32Tensor["S H W 1"]]
    DepthConfTensor: TypeAlias = Annotated[torch.Tensor, dltype.Float32Tensor["S H W"]]
    PointmapTensor: TypeAlias = Annotated[torch.Tensor, dltype.Float32Tensor["S H W 3"]]
    PointmapConfTensor: TypeAlias = Annotated[
        torch.Tensor, dltype.Float32Tensor["S H W"]
    ]

    class UInt8Float32Tensor(dltype.TensorTypeBase):
        def check(self, tensor, tensor_name="anonymous"):
            import torch
            from dltype import DLTypeDtypeError

            super().check(tensor, tensor_name)

            if tensor.dtype not in (torch.uint8, torch.float32):
                raise DLTypeDtypeError(
                    tensor_name=tensor_name,
                    expected=(torch.uint8, torch.float32),
                    received=(tensor.dtype,),
                )

    ImagesTensorLike: TypeAlias = Annotated[torch.Tensor, UInt8Float32Tensor["S 3 H W"]]
    ImagesAlphaTensorLike: TypeAlias = Annotated[
        torch.Tensor, UInt8Float32Tensor["S 1 H W"]
    ]

    O = TypeVar("O")
    S = TypeVar("S")

    @staticmethod
    def annotation_of(annotated: Annotated) -> dltype.TensorTypeBase:
        origin, specifier = get_args(annotated)
        return specifier


@dltype.dltyped_dataclass()
@dataclass(frozen=True, slots=True)
class PointCloudTensors:
    xyz: TensorTypes.XYZTensor
    rgb_0_255: TensorTypes.RGB_0_255_Tensor
    conf: TensorTypes.ConfTensor  # confidence level of each point
    point_cams: TensorTypes.PointCamsTensor  # which camera each point came from

    extrinsic: TensorTypes.ExtrinsicTensor
    intrinsic: TensorTypes.IntrinsicTensor

    depth: TensorTypes.DepthTensor
    depth_conf: TensorTypes.DepthConfTensor
    pointmap: TensorTypes.PointmapTensor
    pointmap_conf: TensorTypes.PointmapConfTensor

    def to_dict(self) -> Dict[str, torch.Tensor]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, torch.Tensor]) -> Self:
        try:
            return cls(**d)
        except (TypeError, ValueError):  # make the error type clear
            raise

    @classmethod
    def map(cls) -> Dict[str, dltype.TensorTypeBase]:
        field_hints = get_type_hints(cls, include_extras=True)
        return {name: get_args(hint)[1] for name, hint in field_hints.items()}

    def to(self, device: torch.Device):
        for fld in fields(self):
            tensor: torch.Tensor = getattr(self, fld.name)
            tensor.to(device)


@dltype.dltyped()
def to_0_1(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float32:
        return tensor

    assert (
        tensor.dtype == torch.uint8
    ), "Given tensor should only ever be of type float32 or uint8"

    return tensor.to(torch.float32).div_(255.0)


@dltype.dltyped()
def to_0_255(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        return tensor

    assert (
        tensor.dtype == torch.float32
    ), "Given tensor should only ever be of type float32 or uint8"

    return (tensor.clamp(0, 1).mul_(255.0)).round().to(torch.uint8)


@dltype.dltyped()
def to_channel_as_primary(tensor: torch.Tensor) -> torch.Tensor:
    assert len(tensor.shape) == 4
    return tensor.permute(0, 3, 1, 2)


@dltype.dltyped()
def to_channel_as_item(tensor: torch.Tensor) -> torch.Tensor:
    assert len(tensor.shape) == 4
    return tensor.permute(0, 2, 3, 1)


@dltype.dltyped()
def save_tensor_stack_png_preview(
    tensor: TensorTypes.ImagesTensor_0_1, tensor_out_file: Path, suffix: str = ""
):
    from torchvision.utils import save_image

    tensor_0_1 = to_0_1(tensor)

    preview_png_file: Path = (
        tensor_out_file.parent / f"{tensor_out_file.stem}{suffix}.png"
    )

    save_image(tensor_0_1, preview_png_file, nrow=4)


@dltype.dltyped()
def save_images_tensor(
    out_file: Path,
    metadata: FrameTensorMetadata,
    images_0_1: TensorTypes.ImagesAlphaTensor_0_1,
    images_alpha_0_1: Optional[TensorTypes.ImagesAlphaTensor_0_1],
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


@dltype.dltyped()
def load_and_verify_tensor_file(
    in_file: Path,
    device: torch.Device,
    new_metadata: FrameTensorMetadata,
    map: Dict[str, dltype.TensorTypeBase],  # keys to type annotations
) -> Dict[str, torch.Tensor]:
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

    tensors: Dict[str, torch.Tensor] = {}
    with safe_open(in_file, framework="pt", device=str(device)) as f:
        file: safe_open = f
        for key, annotation in map.items():
            try:
                tensor = file.get_tensor(key)
                annotation.check(tensor, key)

                tensors |= {key: file.get_tensor(key)}
            except (SafetensorError, dltype.DLTypeError) as e:
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


@dltype.dltyped()
def crop_tensor(
    tensor: Annotated[torch.Tensor, dltype.FloatTensor["B C H W"]],
    crop_geom: CropGeometry,
    *,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> Annotated[torch.Tensor, dltype.FloatTensor["B C H_out W_out"]]:
    import torch.nn.functional as F

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


@dltype.dltyped()
def save_ply_ascii(
    out_file: Path, xyz: TensorTypes.XYZTensor, rgb_0_255: TensorTypes.RGB_0_255_Tensor
):
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


@dltype.dltyped()
def save_ply_binary(
    out_file: Path, xyz: TensorTypes.XYZTensor, rgb_0_255: TensorTypes.RGB_0_255_Tensor
):
    import numpy as np

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
