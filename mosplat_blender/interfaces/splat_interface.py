"""
`M` corresponds to # of splats
`N` corresponds to # of reconstructed points
`K` corresponds to # of SH coefficients per splat. `K = (sh_degree+1)^2)`
`S` corresponds to scene size, which would be # of cameras capturing each frame
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, NamedTuple, Self, Tuple

from ..infrastructure.dl_ops import (
    PointCloudTensors,
    TensorTypes as TT,
    to_0_1,
    to_channel_as_item,
    to_channel_as_primary,
)
from ..infrastructure.schemas import SplatTrainingConfig

if TYPE_CHECKING:
    from jaxtyping import Bool, Float32, Int64
    from torch import Device, Tensor
    from torch.nn import ParameterDict

HASH_MULTIPLIER_X = 73856093
HASH_MULTIPLIER_Y = 19349663
HASH_MULTIPLIER_Z = 83492791

EPS: float = 1e-6


def normalize_quat_tensor_(q: Float32[Tensor, "M 4"]) -> Float32[Tensor, "M 4"]:
    """ensure all quaternion magnitudes of 1. `_` suffix denotes in-place operations"""
    return q.div_(q.norm(dim=-1, keepdim=True).clamp_min(EPS))


def quats_identity(M: int, device: Device) -> Float32[Tensor, "M 4"]:
    """creates tensor of quats where (x, y, z, w) = [0, 0, 0, 1]"""
    import torch

    q: Float32[Tensor, "M 4"] = torch.zeros((M, 4), device=device, dtype=torch.float32)
    q[:, 3] = 1.0
    return q


def w2c_3x4_to_view_4x4(extrinsic: TT.ExtrinsicTensor) -> Float32[Tensor, "S 4 4"]:
    """world-to-camera to view by simply adding homogeneous bottom row [0, 0, 0, 1]"""
    import torch

    S = extrinsic.shape[0]
    v: Float32[Tensor, "S 4 4"] = torch.zeros(
        (S, 4, 4), device=extrinsic.device, dtype=extrinsic.dtype
    )
    v[:, :3, :4] = extrinsic
    v[:, 3, 3] = 1.0
    return v


def sh_from_rgb(
    rgb: Float32[Tensor, "M 3"], sh_degree: int = 0
) -> Float32[Tensor, "M K 3"]:
    """derive spherical harmonics from RGB values, where only the DC (Direct Current) is filled in"""
    import torch

    device = rgb.device

    K = (sh_degree + 1) ** 2
    N = rgb.shape[0]

    sh: Float32[Tensor, "M K 3"] = torch.zeros(
        (N, K, 3), device=device, dtype=torch.float32
    )

    sh[:, 0, :] = rgb  # fill dimensions with RGB values, skipping `K` dimension

    return sh


def scales_from_confidence(
    conf: TT.ConfTensor,
    *,
    base_scale: float,
    scale_mult: float,
) -> Tensor:
    """use confidence values to initialize scale of splats"""
    import torch

    s = base_scale * scale_mult / conf.clamp_min(1e-3)
    return torch.stack([s, s, s], dim=1)


def scalars_from_xyz(
    xyz: Float32[Tensor, "N 3"],
    voxel_size_factor: float = 0.005,  # ~2.0m * 0.005 = 0.01 meters
    base_scale_factor: float = 0.02,  # corresponds to human height of ~2m
) -> Tuple[TT.VoxelTensor, float]:
    """derive initial voxel size & base scale as factors of bounding box diagonal"""
    bbox = xyz.max(dim=0)[0] - xyz.min(dim=0)[0]
    diag = bbox.norm()  # `norm` returns a scalar tensor
    return diag * voxel_size_factor, (diag * base_scale_factor).item()


def fuse_points_by_voxel(
    xyz: TT.XYZTensor,
    rgb: Float32[Tensor, "N 3"],
    conf: TT.ConfTensor,
    voxel_size: TT.VoxelTensor,
) -> Tuple[
    Float32[Tensor, "M 3"],  # means
    Float32[Tensor, "M 3"],  # rgb_fused
    Float32[Tensor, "M"],  # depth_fused
]:
    import torch

    device = xyz.device

    # points to voxel IDs, i.e. map continuous 3D points to integer voxel grid
    ids: Int64[Tensor, "N 3"] = torch.floor(xyz / voxel_size).to(torch.int64)

    # create a 1D hash ID for each voxel
    hx: Int64[Tensor, "N"] = ids[:, 0] * HASH_MULTIPLIER_X
    hy: Int64[Tensor, "N"] = ids[:, 1] * HASH_MULTIPLIER_Y
    hz: Int64[Tensor, "N"] = ids[:, 2] * HASH_MULTIPLIER_Z
    h: Int64[Tensor, "N"] = (hx ^ hy ^ hz).to(torch.int64)

    # define the order using sorted voxel hashes
    order: Int64[Tensor, "N"] = torch.argsort(h)
    # apply order, effectively sorting by voxel so that points in same voxel become contiguous
    h = h[order]
    xyz = xyz[order]
    rgb = rgb[order]
    conf = conf[order]

    # mark boundaries where voxel ID changes, i.e. `True`'s are boundaries b/t voxels
    boundaries: Bool[Tensor, "N"] = torch.ones_like(h, dtype=torch.bool)
    boundaries[1:] = h[1:] != h[:-1]

    # use cumulative sum to map all points to their voxel ID given boundaries
    voxel_ids: Int64[Tensor, "N"] = torch.cumsum(boundaries.to(torch.int64), dim=0) - 1
    M: int = int(voxel_ids[-1].item()) + 1  # num unique voxels / future gaussians

    wt: Float32[Tensor, "N"] = conf.clamp_min(EPS).to(xyz.dtype)  # weight from conf
    wt_sum: Float32[Tensor, "M"] = torch.zeros(
        (M,), device=device, dtype=xyz.dtype
    ).scatter_add_(
        dim=0, index=voxel_ids, src=wt
    )  # scatter based on newfound `voxel_ids`

    # reduce positions using a representative weighted AVERAGE position per voxel
    means: Float32[Tensor, "M 3"] = torch.zeros((M, 3), device=device, dtype=xyz.dtype)
    means.scatter_add_(0, voxel_ids[:, None].expand(-1, 3), xyz * wt[:, None])
    means = means / wt_sum[:, None]

    # reduce rgb using a representative weighted AVERAGE rgb per voxel
    rgb_fused: Float32[Tensor, "M 3"] = torch.zeros(
        (M, 3), device=device, dtype=rgb.dtype
    )
    rgb_fused.scatter_add_(0, voxel_ids[:, None].expand(-1, 3), rgb * wt[:, None])
    rgb_fused = rgb_fused / wt_sum[:, None]  # convert back to 'uint8'

    # reduce confidence using a representative MAX confidence per voxel
    conf_fused: Float32[Tensor, "M"] = (
        torch.zeros((M,), device=device, dtype=conf.dtype)
        .scatter_reduce_(0, voxel_ids, conf, reduce="amax")
        .to(conf.dtype)
    )

    return means, rgb_fused, conf_fused


class SplatModel:
    params: ParameterDict

    def __init__(
        self,
        device,
        *,
        means: Float32[Tensor, "M 3"],
        scales: Float32[Tensor, "M 3"],  # range is > 0
        quats: Float32[Tensor, "M 4"],  # should always be normalized before storage
        opacities: Float32[Tensor, "M 1"],
        sh: Float32[Tensor, "M K 3"],
    ):
        import torch

        super().__init__()

        self.params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(means),
                "scales": torch.nn.Parameter(scales),
                "quats": torch.nn.Parameter(quats),
                "opacities": torch.nn.Parameter(opacities),
                "sh": torch.nn.Parameter(sh),
            }
        )
        for param in self.params.parameters():
            param.to(device)

    @property
    def means(self) -> Float32[Tensor, "M 3"]:
        return self.params.get_parameter("means")

    @property
    def scales(self) -> Float32[Tensor, "M 3"]:
        return self.params.get_parameter("scales")

    @property
    def quats(self) -> Float32[Tensor, "M 4"]:
        return self.params.get_parameter("quats")

    @property
    def opacities(self) -> Float32[Tensor, "M 1"]:
        return self.params.get_parameter("opacities")

    @property
    def sh(self) -> Float32[Tensor, "M K 3"]:
        return self.params.get_parameter("sh")

    def post_step(self):
        import torch

        with torch.no_grad():
            q = self.quats
            normalize_quat_tensor_(q)

    @property
    def detached_tensors(self) -> Tensors:
        return SplatModel.Tensors(
            **{
                name: param.detach().clone()
                for name, param in self.params.named_parameters()
            }
        )

    @classmethod
    def init_from_pointcloud_tensors(
        cls,
        pc_tensors: PointCloudTensors,
        device: Device,
        *,
        voxel_size: TT.VoxelTensor,
        base_scale: float,
        scale_mult: float = 1.0,
    ) -> Self:
        pc_tensors.to(device)  # convert to device
        means, rgb_fused, conf_fused = fuse_points_by_voxel(
            pc_tensors.xyz,
            to_0_1(pc_tensors.rgb_0_255),
            pc_tensors.conf,
            voxel_size=voxel_size,
        )
        M = means.shape[0]

        scales = scales_from_confidence(
            conf_fused, base_scale=base_scale, scale_mult=scale_mult
        )
        quats = normalize_quat_tensor_(quats_identity(M, device))
        opacities = conf_fused.clamp(0.0, 1.0)
        sh = sh_from_rgb(rgb_fused)

        return cls(
            device, means=means, scales=scales, quats=quats, opacities=opacities, sh=sh
        )

    class Tensors(NamedTuple):
        means: Float32[Tensor, "M 3"]
        scales: Float32[Tensor, "M 3"]
        quats: Float32[Tensor, "M 4"]
        opacities: Float32[Tensor, "M 1"]
        sh: Float32[Tensor, "M K 3"]


class GsplatRasterizer:

    def __init__(
        self,
        device: Device,
        image_size: Tuple[int, int],
        sh_degree: int,
        znear: float = 0.01,
        zfar: float = 1000.0,
    ):
        self.H, self.W = image_size
        self.device = device
        self.sh_degree = sh_degree
        self.znear = znear
        self.zfar = zfar

    def render(
        self,
        model: SplatModel,
        extrinsic: TT.ExtrinsicTensor,
        intrinsic: TT.IntrinsicTensor,
        backgrounds: TT.ImagesAlphaTensor_0_1,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        from gsplat import rasterization

        assert model.sh.shape[1] == (self.sh_degree + 1) ** 2

        device = self.device

        extrinsic = extrinsic.to(device)
        intrinsic = intrinsic.to(device)
        backgrounds = backgrounds.to(device)

        view = w2c_3x4_to_view_4x4(extrinsic)

        render_colors, render_alphas, meta = rasterization(
            means=model.means,
            quats=model.quats,
            scales=model.scales,
            opacities=model.opacities,
            colors=model.sh,
            viewmats=view,
            Ks=intrinsic,
            width=self.W,
            height=self.H,
            near_plane=self.znear,
            far_plane=self.zfar,
            sh_degree=self.sh_degree,
            backgrounds=to_channel_as_item(backgrounds),
            render_mode="RGB+ED",
        )

        # render_colors are (S,H,W,4) in `RGB+ED`, where last dimension is RGB + depth
        rgb_as_item: Float32[Tensor, "S H W 3"] = render_colors[..., :3]
        depth_as_item: Float32[Tensor, "S H W 1"] = render_colors[..., 3:]

        alpha_as_item: Float32[Tensor, "S H W 1"] = render_alphas

        rgb = to_channel_as_primary(rgb_as_item)
        depth = to_channel_as_primary(depth_as_item)
        alpha = to_channel_as_primary(alpha_as_item)

        return (rgb, depth, alpha)


def save_ply_3dgs_binary(
    out_file: Path,
    tensors: SplatModel.Tensors,
    encode_log_of_scale: bool = True,  # convert 'scale' as 'log(scale)'
    encode_opacity_in_logit: bool = True,  # convert 'opacity' 0-1 to unbounded logit space
) -> None:
    import numpy as np
    import torch

    means, scales, quats, opacities, sh = tensors

    if encode_log_of_scale:
        scales = scales.log()

    if encode_opacity_in_logit:
        x = opacities.clamp(EPS, 1.0 - EPS)
        opacities = torch.log(x) - torch.log1p(-x)

    M, K, C = sh.shape

    # separate DC and rest
    f_dc = sh[:, 0, :]  # (M,3)
    f_rest = sh[:, 1:, :].reshape(M, -1)  # (M, 3*(K-1))

    props = [
        ("x", "float"),
        ("y", "float"),
        ("z", "float"),
        ("nx", "float"),
        ("ny", "float"),
        ("nz", "float"),
    ]

    for i in range(3):  # DC fields
        props.append((f"f_dc_{i}", "float"))

    num_rest = f_rest.shape[1]
    for i in range(num_rest):  # rest of SH fields
        props.append((f"f_rest_{i}", "float"))

    props += [
        ("opacity", "float"),
        ("scale_0", "float"),
        ("scale_1", "float"),
        ("scale_2", "float"),
        ("rot_0", "float"),
        ("rot_1", "float"),
        ("rot_2", "float"),
        ("rot_3", "float"),
    ]

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {M}",
    ]
    header += [f"property {t} {n}" for (n, t) in props]
    header += ["end_header\n"]
    header_str = "\n".join(header)

    normals = torch.zeros((M, 3), dtype=torch.float32)  # set normals to zero

    body = (
        torch.cat(
            [
                means,
                normals,
                f_dc,
                f_rest,
                opacities,
                scales,
                quats,
            ],
            dim=1,
        )
        .cpu()  # convert to cpu first
        .numpy()
    )

    with out_file.open(mode="wb") as f:
        f.write(header_str.encode("ascii"))
        f.write(body.astype(np.float32).tobytes())


def train_3dgs(
    model: SplatModel,
    rasterizer: GsplatRasterizer,
    pc_tensors: PointCloudTensors,
    *,
    images_0_255: TT.ImagesTensor_0_255,
    images_alpha_0_255: TT.ImagesAlphaTensor_0_255,
    out_file: Path,
    config: SplatTrainingConfig = SplatTrainingConfig(),
) -> SplatModel:
    import torch
    from gsplat import DefaultStrategy

    device = rasterizer.device

    pc_tensors.to(device)
    extrinsic = pc_tensors.extrinsic
    intrinsic = pc_tensors.intrinsic
    depth = pc_tensors.depth
    depth_conf = pc_tensors.depth_conf
    depth_conf_unsqueezed = depth_conf.unsqueeze(1)

    images = to_0_1(images_0_255)
    images_alpha = to_0_1(images_alpha_0_255)

    optimizers: Dict[str, torch.optim.Optimizer] = {
        name: torch.optim.Adam([param], lr=config.lr)
        for name, param in model.params.items()
    }  # gsplat expects one optimizer per parameter

    strategy = DefaultStrategy()
    strategy.check_sanity(model.params, optimizers)
    strategy_state = strategy.initialize_state()

    S: int = images.shape[0]

    for step in range(config.steps):
        idx = torch.randint(0, S, (config.batch_size,), device=device)

        rgb = images.index_select(0, idx)
        alpha = images_alpha.index_select(0, idx)
        extri = extrinsic.index_select(0, idx)
        intri = intrinsic.index_select(0, idx)
        dep = depth.index_select(0, idx)
        dep_cf = depth_conf_unsqueezed.index_select(0, idx)

        bg = torch.zeros(
            (config.batch_size, 3, rasterizer.H, rasterizer.W),
            device=device,
            dtype=torch.float32,
        )

        pred_rgb, pred_depth, pred_alpha = rasterizer.render(model, extri, intri, bg)

        diff = (pred_rgb - rgb).abs()

        rgb_loss = (diff * alpha).sum() / (alpha.sum() + EPS)
        alpha_loss = ((pred_alpha - alpha).abs()).mean()

        depth_diff = (pred_depth - dep).abs()
        depth_mask = alpha * dep_cf  # use depth confidence as extra stabilizer

        depth_loss = (depth_diff * depth_mask).sum() / (depth_mask.sum() + EPS)

        loss = (
            rgb_loss
            + config.alpha_weight * alpha_loss
            + config.depth_weight * depth_loss
        )

        strategy.step_pre_backward(
            model.params,
            optimizers,
            strategy_state,
            step,
            {},
        )

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)

        loss.backward()

        strategy.step_post_backward(
            model.params,
            optimizers,
            strategy_state,
            step,
            {},
        )

        for opt in optimizers.values():
            opt.step()

        model.post_step()

        if step % 1000 == 0:
            with torch.no_grad():
                save_ply_3dgs_binary(out_file, model.detached_tensors)

    return model
