"""deep-learning operations utilizing PyTorch tensors. should use nested import"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import (
    Annotated as Anno,
    Any,
    Dict,
    NamedTuple,
    Optional,
    Self,
    Tuple,
    TypeAlias,
    get_args,
    get_type_hints,
)

import dltype
import torch

from .constants import VGGT_IMAGE_DIMS_FACTOR, VGGT_MAX_IMAGE_SIZE
from .macros import add_suffix_to_path, try_access_path
from .schemas import (
    CropGeometry,
    ExportedTensorKey,
    FrameTensorMetadata,
    UserAssertionError,
)


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


class TensorTypes(SimpleNamespace):
    ImagesTensor_0_1: TypeAlias = Anno[torch.Tensor, dltype.Float32Tensor["S 3 H W"]]
    ImagesTensor_0_255: TypeAlias = Anno[torch.Tensor, dltype.UInt8Tensor["S 3 H W"]]

    ImagesAlphaTensor_0_1: TypeAlias = Anno[
        torch.Tensor, dltype.Float32Tensor["S 1 H W"]
    ]
    ImagesAlphaTensor_0_255: TypeAlias = Anno[
        torch.Tensor, dltype.UInt8Tensor["S 1 H W"]
    ]

    VoxelTensor: TypeAlias = Anno[torch.Tensor, dltype.Float32Tensor[None]]

    XYZTensor: TypeAlias = Anno[torch.Tensor, dltype.Float32Tensor["N 3"]]
    RGB_0_255_Tensor: TypeAlias = Anno[torch.Tensor, dltype.UInt8Tensor["N 3"]]
    ConfTensor: TypeAlias = Anno[torch.Tensor, dltype.Float32Tensor["N"]]
    PointCamsTensor: TypeAlias = Anno[torch.Tensor, dltype.Int32Tensor["N"]]

    ExtrinsicTensor: TypeAlias = Anno[torch.Tensor, dltype.Float32Tensor["S 3 4"]]
    IntrinsicTensor: TypeAlias = Anno[torch.Tensor, dltype.Float32Tensor["S 3 3"]]

    DepthTensor: TypeAlias = Anno[torch.Tensor, dltype.Float32Tensor["S H W 1"]]
    DepthConfTensor: TypeAlias = Anno[torch.Tensor, dltype.Float32Tensor["S H W"]]
    PointmapTensor: TypeAlias = Anno[torch.Tensor, dltype.Float32Tensor["S H W 3"]]
    PointmapConfTensor: TypeAlias = Anno[torch.Tensor, dltype.Float32Tensor["S H W"]]

    ImagesTensorLike: TypeAlias = Anno[torch.Tensor, UInt8Float32Tensor["S 3 H W"]]
    ImagesAlphaTensorLike: TypeAlias = Anno[torch.Tensor, UInt8Float32Tensor["S 1 H W"]]

    @staticmethod
    def annotation_of(annotated: Anno) -> dltype.TensorTypeBase:
        origin, specifier = get_args(annotated)
        return specifier

    @classmethod
    def raw_annotation_map(cls):
        return {
            ExportedTensorKey.IMAGES.value: cls.annotation_of(cls.ImagesTensor_0_255)
        }

    @classmethod
    def preprocessed_annotation_map(cls):
        return {
            ExportedTensorKey.IMAGES.value: cls.annotation_of(cls.ImagesTensor_0_255),
            ExportedTensorKey.IMAGES_ALPHA.value: cls.annotation_of(
                cls.ImagesAlphaTensor_0_255
            ),
        }


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
    def annotation_map(cls) -> Dict[str, dltype.TensorTypeBase]:
        field_hints = get_type_hints(cls, include_extras=True)
        return {name: get_args(hint)[1] for name, hint in field_hints.items()}

    def to(self, device: torch.device):
        for fld in fields(self):
            tensor: torch.Tensor = getattr(self, fld.name)
            tensor.to(device)


class VGGTPredictions(NamedTuple):
    pose_enc: Anno[torch.Tensor, dltype.Float32Tensor["B S 9"]]
    depth: Anno[torch.Tensor, dltype.Float32Tensor["B S H W 1"]]
    depth_conf: Anno[torch.Tensor, dltype.Float32Tensor["B S H W"]]
    world_points: Anno[torch.Tensor, dltype.Float32Tensor["B S H W 3"]]
    world_points_conf: Anno[torch.Tensor, dltype.Float32Tensor["B S H W"]]
    images: TensorTypes.ImagesAlphaTensor_0_1

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Self:
        """create conservatively by parsing away undocumented fields"""
        parsed = {k: v for k, v in d.items() if k in cls._fields}
        return cls(**parsed)


@dltype.dltyped()
def to_0_1(
    tensor: Anno[torch.Tensor, UInt8Float32Tensor["*dims"]],
) -> Anno[torch.Tensor, dltype.Float32Tensor["*dims"]]:
    if tensor.dtype == torch.float32:
        return tensor

    assert (
        tensor.dtype == torch.uint8
    ), "Given tensor should only ever be of type float32 or uint8"

    return tensor.to(torch.float32).div_(255.0)


@dltype.dltyped()
def to_0_255(
    tensor: Anno[torch.Tensor, UInt8Float32Tensor["*dims"]],
) -> Anno[torch.Tensor, dltype.UInt8Tensor["*dims"]]:
    if tensor.dtype == torch.uint8:
        return tensor

    assert (
        tensor.dtype == torch.float32
    ), "Given tensor should only ever be of type float32 or uint8"

    return (tensor.clamp(0, 1).mul_(255.0)).round().to(torch.uint8)


@dltype.dltyped()
def to_channel_as_primary(
    tensor: Anno[torch.Tensor, dltype.TensorTypeBase["S H W C"]],
) -> Anno[torch.Tensor, dltype.TensorTypeBase["S C H W"]]:
    return tensor.permute(0, 3, 1, 2)


@dltype.dltyped()
def to_channel_as_item(
    tensor: Anno[torch.Tensor, dltype.TensorTypeBase["S C H W"]],
) -> Anno[torch.Tensor, dltype.TensorTypeBase["S H W C"]]:
    return tensor.permute(0, 2, 3, 1)


@dltype.dltyped()
def save_images_png_preview(
    images: TensorTypes.ImagesTensorLike, tensor_out_file: Path, suffix: str = ""
):
    from torchvision.utils import save_image

    images_0_1 = to_0_1(images)

    preview_png_file = add_suffix_to_path(tensor_out_file, suffix)

    save_image(images_0_1, preview_png_file, nrow=4)


@dltype.dltyped()
def save_images_safetensors(
    out_file: Path,
    metadata: FrameTensorMetadata,
    images: TensorTypes.ImagesTensorLike,
    images_alpha: Optional[Anno[torch.Tensor, UInt8Float32Tensor["S 1 H W"]]],
):
    from safetensors.torch import save_file

    images_0_255 = to_0_255(images)
    out_tensors = {ExportedTensorKey.IMAGES.value: images_0_255}

    if images_alpha is not None:
        images_alpha_0_255 = to_0_255(images_alpha)
        out_tensors |= {ExportedTensorKey.IMAGES_ALPHA.value: images_alpha_0_255}

    save_file(
        out_tensors,
        filename=out_file,
        metadata=metadata.to_dict(),
    )


def load_safetensors(
    in_file: Path,
    device: torch.device,
    expected_metadata: FrameTensorMetadata,
    annotation_map: Dict[str, dltype.TensorTypeBase],  # keys to type annotations
) -> Dict[str, torch.Tensor]:
    from safetensors import SafetensorError, safe_open

    from .schemas import (
        FrameTensorMetadata,
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
        for key, annotation in annotation_map.items():
            try:
                tensor = file.get_tensor(key)
                annotation.check(tensor, key)

                tensors |= {key: file.get_tensor(key)}
            except (SafetensorError, dltype.DLTypeError) as e:
                raise UserFacingError from e

        try:
            metadata: FrameTensorMetadata = FrameTensorMetadata.from_dict(
                file.metadata()
            )
        except TypeError as e:
            raise UserFacingError(
                f"Metadata used to create '{in_file}' does not match the structure of new desired metadata.",
                e,
            ) from e

    expected_metadata.compare(metadata)  # raises `UserAssertionError` on failure

    return tensors


@dltype.dltyped()
def crop_tensor(
    tensor: Anno[torch.Tensor, dltype.FloatTensor["B C H W"]],
    crop_geom: CropGeometry,
    *,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> Anno[torch.Tensor, dltype.FloatTensor["B C H_cropped W_cropped"]]:
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


@dltype.dltyped()
def ensure_tensor_inputs_for_vggt(
    images: TensorTypes.ImagesTensorLike,
    images_alpha: TensorTypes.ImagesAlphaTensorLike,
):
    def _ensure_tensor_shape_for_vggt(tensor):
        """ensure tensors are the correct shape for VGGT model"""
        _, _, H, W = tensor.shape
        assert H <= VGGT_MAX_IMAGE_SIZE and W <= VGGT_MAX_IMAGE_SIZE
        assert H % VGGT_IMAGE_DIMS_FACTOR == 0 and W % VGGT_IMAGE_DIMS_FACTOR == 0

    """uses `dltype` for shape and type-checking, then ensure other constraints are met"""
    _ensure_tensor_shape_for_vggt(images)
    _ensure_tensor_shape_for_vggt(images_alpha)


@dltype.dltyped()
def validate_preprocess_script_output(
    output, _: TensorTypes.ImagesTensor_0_1
) -> Tuple[TensorTypes.ImagesTensor_0_1, TensorTypes.ImagesAlphaTensor_0_1]:
    if not isinstance(output, tuple):
        raise UserAssertionError(
            f"Return value of preprocess script must be a tuple",
            expected=tuple.__name__,
            actual=type(output).__name__,
        )
    if not len(output) == 2:
        raise UserAssertionError(
            f"Return value of preprocess script must be a tuple of size 2",
            expected=len(output),
            actual=2,
        )

    images, images_alpha = output

    return images, images_alpha
