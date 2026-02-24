"""
`M` corresponds to # of splats
`N` corresponds to # of reconstructed points
`K` corresponds to # of SH coefficients per splat. `K = (sh_degree+1)^2)`
`S` corresponds to scene size, which would be # of cameras capturing each frame

try to keep as lazy import as there is top-level import of `dltype` and `torch`
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated as Anno, Dict, NamedTuple, Self, Tuple

from ..infrastructure.dl_ops import (
    PointCloudTensors,
    TensorTypes as TT,
    UInt8Float32Tensor,
    to_0_1,
    to_channel_as_item,
    to_channel_as_primary,
)
from ..infrastructure.schemas import SplatTrainingConfig

HASH_MULTIPLIER_X = 73856093
HASH_MULTIPLIER_Y = 19349663
HASH_MULTIPLIER_Z = 83492791

EPS: float = 1e-6

import dltype
import torch


@dltype.dltyped()
def normalize_quat_tensor_(
    q: Anno[torch.Tensor, dltype.Float32Tensor["M 4"]],
) -> Anno[torch.Tensor, dltype.Float32Tensor["M 4"]]:
    """ensure all quaternion magnitudes of 1. `_` suffix denotes in-place operations"""
    return q.div_(q.norm(dim=-1, keepdim=True).clamp_min(EPS))


@dltype.dltyped()
def quats_identity(
    M: int, device: torch.device
) -> Anno[torch.Tensor, dltype.Float32Tensor["M 4"]]:
    """creates tensor of quats where (x, y, z, w) = [0, 0, 0, 1]"""

    q: Anno[torch.Tensor, dltype.Float32Tensor["M 4"]] = torch.zeros(
        (M, 4), device=device, dtype=torch.float32
    )
    q[:, 3] = 1.0
    return q


@dltype.dltyped()
def w2c_3x4_to_view_4x4(
    extrinsic: TT.ExtrinsicTensor,
) -> Anno[torch.Tensor, dltype.Float32Tensor["S 4 4"]]:
    """world-to-camera to view by simply adding homogeneous bottom row [0, 0, 0, 1]"""

    S = extrinsic.shape[0]
    v: Anno[torch.Tensor, dltype.Float32Tensor["S 4 4"]] = torch.zeros(
        (S, 4, 4), device=extrinsic.device, dtype=extrinsic.dtype
    )
    v[:, :3, :4] = extrinsic
    v[:, 3, 3] = 1.0
    return v


@dltype.dltyped()
def sh_from_rgb(
    rgb: Anno[torch.Tensor, UInt8Float32Tensor["M 3"]], sh_degree
) -> Anno[torch.Tensor, dltype.Float32Tensor["M K 3"]]:
    """derive spherical harmonics from RGB values, where only the DC (Direct Current) is filled in"""

    device = rgb.device

    rgb_0_1 = to_0_1(rgb)

    K = (sh_degree + 1) ** 2
    N = rgb.shape[0]

    sh: Anno[torch.Tensor, dltype.Float32Tensor["M K 3"]] = torch.zeros(
        (N, K, 3), device=device, dtype=torch.float32
    )

    sh[:, 0, :] = rgb_0_1  # fill dimensions with RGB values, skipping `K` dimension

    return sh


@dltype.dltyped()
def scales_from_confidence(
    conf: TT.ConfTensor,
    *,
    base_scale: float,
    scale_mult: float,
) -> Anno[torch.Tensor, dltype.Float32Tensor["N 3"]]:
    """use confidence values to initialize scale of splats"""

    conf_median = conf.median()

    scale_factor = conf_median / (conf + EPS)

    s = base_scale * scale_mult * scale_factor
    return torch.stack([s, s, s], dim=1)


@dltype.dltyped()
def scalars_from_xyz(
    xyz: Anno[torch.Tensor, dltype.Float32Tensor["N 3"]],
    voxel_size_factor: float = 0.005,  # ~2.0m * 0.005 = 0.01 meters
    base_scale_factor: float = 0.02,  # corresponds to human height of ~2m
) -> Tuple[TT.VoxelTensor, float]:
    """derive initial voxel size & base scale as factors of bounding box diagonal"""
    bbox = xyz.max(dim=0)[0] - xyz.min(dim=0)[0]
    diag = bbox.norm()  # `norm` returns a scalar tensor
    return diag * voxel_size_factor, (diag * base_scale_factor).item()


@dltype.dltyped()
def fuse_points_by_voxel(
    xyz: TT.XYZTensor,
    rgb_0_1: Anno[torch.Tensor, dltype.Float32Tensor["N 3"]],
    conf: TT.ConfTensor,
    voxel_size: TT.VoxelTensor,
) -> Tuple[
    Anno[torch.Tensor, dltype.Float32Tensor["M 3"]],  # means
    Anno[torch.Tensor, dltype.Float32Tensor["M 3"]],  # rgb_fused
    Anno[torch.Tensor, dltype.Float32Tensor["M"]],  # conf_fused
]:

    device = xyz.device

    # points to voxel IDs, i.e. map continuous 3D points to integer voxel grid
    ids: Anno[torch.Tensor, dltype.Int64Tensor["N 3"]] = torch.floor(
        xyz / voxel_size
    ).to(torch.int64)

    # create a 1D hash ID for each voxel
    hx: Anno[torch.Tensor, dltype.Int64Tensor["N"]] = ids[:, 0] * HASH_MULTIPLIER_X
    hy: Anno[torch.Tensor, dltype.Int64Tensor["N"]] = ids[:, 1] * HASH_MULTIPLIER_Y
    hz: Anno[torch.Tensor, dltype.Int64Tensor["N"]] = ids[:, 2] * HASH_MULTIPLIER_Z
    h: Anno[torch.Tensor, dltype.Int64Tensor["N"]] = (hx ^ hy ^ hz).to(torch.int64)

    # define the order using sorted voxel hashes
    order: Anno[torch.Tensor, dltype.Int64Tensor["N"]] = torch.argsort(h)
    # apply order, effectively sorting by voxel so that points in same voxel become contiguous
    h = h[order]
    xyz = xyz[order]
    rgb_0_1 = rgb_0_1[order]
    conf = conf[order]

    # mark boundaries where voxel ID changes, i.e. `True`'s are boundaries b/t voxels
    boundaries: Anno[torch.Tensor, dltype.BoolTensor["N"]] = torch.ones_like(
        h, dtype=torch.bool
    )
    boundaries[1:] = h[1:] != h[:-1]

    # use cumulative sum to map all points to their voxel ID given boundaries
    voxel_ids: Anno[torch.Tensor, dltype.Int64Tensor["N"]] = (
        torch.cumsum(boundaries.to(torch.int64), dim=0) - 1
    )
    M: int = int(voxel_ids[-1].item()) + 1  # num unique voxels / future gaussians

    wt: Anno[torch.Tensor, dltype.Float32Tensor["N"]] = conf.clamp_min(EPS).to(
        xyz.dtype
    )  # weight from conf
    wt_sum: Anno[torch.Tensor, dltype.Float32Tensor["M"]] = torch.zeros(
        (M,), device=device, dtype=xyz.dtype
    ).scatter_add_(
        dim=0, index=voxel_ids, src=wt
    )  # scatter based on newfound `voxel_ids`

    # reduce positions using a representative weighted AVERAGE position per voxel
    means: Anno[torch.Tensor, dltype.Float32Tensor["M 3"]] = torch.zeros(
        (M, 3), device=device, dtype=xyz.dtype
    )
    means.scatter_add_(0, voxel_ids[:, None].expand(-1, 3), xyz * wt[:, None])
    means = means / wt_sum[:, None]

    # reduce rgb using a representative weighted AVERAGE rgb per voxel
    rgb_fused: Anno[torch.Tensor, dltype.Float32Tensor["M 3"]] = torch.zeros(
        (M, 3), device=device, dtype=rgb_0_1.dtype
    )
    rgb_fused.scatter_add_(0, voxel_ids[:, None].expand(-1, 3), rgb_0_1 * wt[:, None])
    rgb_fused = rgb_fused / wt_sum[:, None]  # convert back to 'uint8'

    # reduce confidence using a representative MAX confidence per voxel
    conf_fused: Anno[torch.Tensor, dltype.Float32Tensor["M"]] = (
        torch.zeros((M,), device=device, dtype=conf.dtype)
        .scatter_reduce_(0, voxel_ids, conf, reduce="amax")
        .to(conf.dtype)
    )

    return means, rgb_fused, conf_fused


class SplatModel(torch.nn.Module):
    @dltype.dltyped()
    def __init__(
        self,
        device,
        *,
        means: Anno[torch.Tensor, dltype.Float32Tensor["M 3"]],
        scales: Anno[torch.Tensor, dltype.Float32Tensor["M 3"]],  # range is > 0
        quats: Anno[
            torch.Tensor, dltype.Float32Tensor["M 4"]
        ],  # should always be normalized before storage
        opacities: Anno[torch.Tensor, dltype.Float32Tensor["M"]],
        sh: Anno[torch.Tensor, dltype.Float32Tensor["M K 3"]],
    ):

        super().__init__()

        self.params: torch.nn.ParameterDict = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(means),
                "scales": torch.nn.Parameter(scales),
                "quats": torch.nn.Parameter(quats),
                "opacities": torch.nn.Parameter(opacities),
                "sh": torch.nn.Parameter(sh),
            }
        )
        for param in self.parameters():
            param.to(device)

    @property
    def means_(self) -> Anno[torch.Tensor, dltype.Float32Tensor["M 3"]]:
        return self.params["means"]

    @property
    def scales_(self) -> Anno[torch.Tensor, dltype.Float32Tensor["M 3"]]:
        return self.params["scales"]

    @property
    def quats_(self) -> Anno[torch.Tensor, dltype.Float32Tensor["M 4"]]:
        return self.params["quats"]

    @property
    def opacities_(self) -> Anno[torch.Tensor, dltype.Float32Tensor["M"]]:
        return self.params["opacities"]

    @property
    def sh_(self) -> Anno[torch.Tensor, dltype.Float32Tensor["M K 3"]]:
        return self.params["sh"]

    def post_step(self):
        with torch.no_grad():
            q = self.quats_
            normalize_quat_tensor_(q)

            self.scales_.data.clamp_(min=1e-3)  # prevent shrink to 0
            self.opacities_.data.clamp_(min=0.05, max=0.99)  # prevent opacity collapse

    @property
    def detached_tensors(self) -> ModelTensors:
        return self.ModelTensors(
            **{
                name: param.detach().clone()
                for name, param in self.params.named_parameters()
            }
        )

    @classmethod
    @dltype.dltyped()
    def init_from_pointcloud_tensors(
        cls,
        pct: PointCloudTensors,
        device: torch.device,
        *,
        voxel_size: TT.VoxelTensor,
        base_scale: float,
        sh_degree: int = 0,
        scale_mult: float = 1.0,
    ) -> Self:
        pct.to(device)  # convert to device
        means, rgb_fused, conf_fused = fuse_points_by_voxel(
            pct.xyz, to_0_1(pct.rgb_0_255), pct.conf, voxel_size=voxel_size
        )
        M = means.shape[0]

        scales = scales_from_confidence(
            conf_fused, base_scale=base_scale, scale_mult=scale_mult
        )
        quats = normalize_quat_tensor_(quats_identity(M, device))
        opacities: Anno[torch.Tensor, dltype.Float32Tensor["M"]] = conf_fused.clamp(
            0.0, 1.0
        )
        sh = sh_from_rgb(rgb_fused, sh_degree)

        return cls(
            device, means=means, scales=scales, quats=quats, opacities=opacities, sh=sh
        )

    @dltype.dltyped_namedtuple()
    class ModelTensors(NamedTuple):
        means: Anno[torch.Tensor, dltype.Float32Tensor["M 3"]]
        scales: Anno[torch.Tensor, dltype.Float32Tensor["M 3"]]
        quats: Anno[torch.Tensor, dltype.Float32Tensor["M 4"]]
        opacities: Anno[torch.Tensor, dltype.Float32Tensor["M"]]
        sh: Anno[torch.Tensor, dltype.Float32Tensor["M K 3"]]


class GsplatRasterizer:
    def __init__(
        self,
        device: torch.device,
        image_HW: Tuple[int, int],
        *,
        sh_degree: int = 0,
        znear: float = 0.01,
        zfar: float = 1000.0,
    ):
        self.H, self.W = image_HW
        self.device = device
        self.sh_degree = sh_degree
        self.znear = znear
        self.zfar = zfar

    @dltype.dltyped()
    def render(
        self,
        model: SplatModel,
        extrinsic: TT.ExtrinsicTensor,
        intrinsic: TT.IntrinsicTensor,
        backgrounds: Anno[torch.Tensor, dltype.Float32Tensor["S 3"]],
    ) -> Tuple[
        Anno[torch.Tensor, dltype.Float32Tensor["S 3 H W"]],
        Anno[torch.Tensor, dltype.Float32Tensor["S 1 H W"]],
        Anno[torch.Tensor, dltype.Float32Tensor["S 1 H W"]],
        Dict,
    ]:
        from gsplat import rasterization

        assert model.sh_.shape[1] == (self.sh_degree + 1) ** 2

        device = self.device

        extrinsic = extrinsic.to(device)
        intrinsic = intrinsic.to(device)
        backgrounds = backgrounds.to(device)

        view = w2c_3x4_to_view_4x4(extrinsic)

        render_colors, render_alphas, meta = rasterization(
            means=model.means_,
            quats=model.quats_,
            scales=model.scales_,
            opacities=model.opacities_,
            colors=model.sh_,
            viewmats=view,
            Ks=intrinsic,
            width=self.W,
            height=self.H,
            near_plane=self.znear,
            far_plane=self.zfar,
            sh_degree=self.sh_degree,
            packed=True,
            render_mode="RGB+ED",
            absgrad=True,
        )

        # render_colors are (S,H,W,4) in `RGB+ED`, where last dimension is RGB + depth
        rgb_as_item: Anno[torch.Tensor, dltype.Float32Tensor["S H W 3"]] = (
            render_colors[..., :3]
        )
        depth_as_item: Anno[torch.Tensor, dltype.Float32Tensor["S H W 1"]] = (
            render_colors[..., 3:]
        )

        alpha_as_item: Anno[torch.Tensor, dltype.Float32Tensor["S H W 1"]] = (
            render_alphas
        )

        rgb: Anno[torch.Tensor, dltype.Float32Tensor["S 3 H W"]] = (
            to_channel_as_primary(rgb_as_item)
        )
        depth: Anno[torch.Tensor, dltype.Float32Tensor["S 1 H W"]] = (
            to_channel_as_primary(depth_as_item)
        )
        alpha: Anno[torch.Tensor, dltype.Float32Tensor["S 1 H W"]] = (
            to_channel_as_primary(alpha_as_item)
        )

        return (rgb, depth, alpha, meta)


def save_ply_3dgs_binary(
    out_file: Path,
    tensors: SplatModel.ModelTensors,
    encode_log_of_scale: bool = True,  # convert 'scale' as 'log(scale)'
    encode_opacity_in_logit: bool = True,  # convert 'opacity' 0-1 to unbounded logit space
):
    import numpy as np

    means, scales, quats, opacities, sh = tensors

    if encode_log_of_scale:
        scales = scales.log()

    if encode_opacity_in_logit:
        x = opacities.clamp(EPS, 1.0 - EPS)
        opacities = torch.log(x) - torch.log1p(-x)

    opacities_unsqueezed: Anno[torch.Tensor, dltype.Float32Tensor["M 1"]] = (
        opacities.unsqueeze(dim=1)
    )

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

    normals = torch.zeros(
        (M, 3), dtype=torch.float32, device=means.device
    )  # set normals to zero

    body = (
        torch.cat(
            [means, normals, f_dc, f_rest, opacities_unsqueezed, scales, quats], dim=1
        )
        .cpu()
        .numpy()
    )

    with out_file.open(mode="wb") as f:
        f.write(header_str.encode("ascii"))
        f.write(body.astype(np.float32).tobytes())


@dltype.dltyped()
def train_3dgs(
    model: SplatModel,
    rasterizer: GsplatRasterizer,
    pct: PointCloudTensors,
    images: TT.ImagesTensorLike,
    images_alpha: TT.ImagesAlphaTensorLike,
    ply_out_file: Path,
    config: SplatTrainingConfig,
) -> SplatModel:
    from gsplat import DefaultStrategy

    device = rasterizer.device

    pct.to(device)
    extrinsic = pct.extrinsic
    intrinsic = pct.intrinsic
    depth = to_channel_as_primary(pct.depth)
    depth_conf = pct.depth_conf.unsqueeze(1)

    images_0_1 = to_0_1(images)
    images_alpha_0_1 = to_0_1(images_alpha)

    optimizers: Dict[str, torch.optim.Optimizer] = {
        name: torch.optim.Adam([param], lr=config.lr)
        for name, param in model.params.items()
    }  # gsplat expects one optimizer per parameter

    strategy = DefaultStrategy()
    strategy.check_sanity(model.params, optimizers)
    strategy_state = strategy.initialize_state()

    S: int = images_0_1.shape[0]

    for step in range(config.steps):
        idx = torch.randint(0, S, (config.scene_size,), device=device)

        rgb = images_0_1.index_select(0, idx)
        alp = images_alpha_0_1.index_select(0, idx)
        extri = extrinsic.index_select(0, idx)
        intri = intrinsic.index_select(0, idx)
        dep = depth.index_select(0, idx)
        dep_cf = depth_conf.index_select(0, idx)

        bg = torch.zeros(
            (config.scene_size, 3),
            device=device,
            dtype=torch.float32,
        )

        pred_rgb, pred_depth, pred_alpha, meta = rasterizer.render(
            model, extri, intri, bg
        )

        mask = alp > 0.5

        if mask.sum() == 0:
            continue  # if no foreground pixels in batch, skip step

        rgb_loss = (pred_rgb - rgb).abs()[mask].mean()
        alpha_loss = (pred_alpha - alp).abs()[mask].mean()
        depth_loss = (pred_depth - dep).abs()[mask].mean()

        loss = (
            rgb_loss
            + config.alpha_weight * alpha_loss
            + config.depth_weight * depth_loss
        )

        strategy.step_pre_backward(model.params, optimizers, strategy_state, step, meta)

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)

        loss.backward()

        strategy.step_post_backward(
            model.params, optimizers, strategy_state, step, meta, packed=True
        )

        for opt in optimizers.values():
            opt.step()

        model.post_step()

        if step % config.save_ply_interval == 0 or step == config.steps - 1:
            with torch.no_grad():
                save_ply_3dgs_binary(ply_out_file, model.detached_tensors)

    return model
