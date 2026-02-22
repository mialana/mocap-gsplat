"""
`M` corresponds to # of splats
`N` corresponds to # of reconstructed points
`K` corresponds to # of SH coefficients per splat. `K = (sh_degree+1)^2)`
`B` corresponds to batch size, which would be # of cameras per frame
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Self, Tuple, TypeAlias, Union

# since there is a top-level `torch` import, try to always lazy import
from torch.nn import Module

from ..infrastructure.macros import to_0_1
from ..infrastructure.schemas import PointCloudTensors, TensorTypes as TT

if TYPE_CHECKING:
    from jaxtyping import Bool, Float32, Int64, UInt8
    from torch import Tensor, device
    from torch.nn import ParameterDict

    VoxelTensor: TypeAlias = Float32[Tensor, ""]

HASH_MULTIPLIER_X = 73856093
HASH_MULTIPLIER_Y = 19349663
HASH_MULTIPLIER_Z = 83492791

EPS: float = 1e-6


@dataclass
class TrainConfig:
    steps: int = 30000
    lr: float = 1e-2  # learning rate (`EPS` is too slow)
    sh_degree: int = 0
    cam_batch: int = 1  # cameras per frame

    save_ply_interval: int = 5000  # i.e. every x steps


@dataclass
class SplatModel:
    params: ParameterDict

    def __init__(
        self,
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

    def detached(self) -> Dict[str, Tensor]:
        return {k: v.detach() for k, v in self.params.named_parameters()}

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

    def normalize_quats(self):
        import torch

        with torch.no_grad():
            q = self.params.get_parameter("quats")
            normalize_quat_tensor_(q)

    @classmethod
    def init_from_pointcloud(
        cls,
        *,
        xyz: TT.XYZTensor,
        rgb: TT.RGBTensorLike,
        conf: TT.ConfTensor,
        voxel_size: VoxelTensor,
        base_scale: float,
        scale_mult: float = 1.0,
    ) -> Self:
        import torch

        device = xyz.device

        means, rgb_fused, conf_fused = fuse_points_by_voxel(
            xyz, rgb, conf, voxel_size=voxel_size
        )
        M = means.shape[0]

        scales = scales_from_confidence(
            conf_fused, base_scale=base_scale, scale_mult=scale_mult
        ).to(device)
        quats = normalize_quat_tensor_(quats_identity(M, device))
        opacities = conf_fused.clamp(0.0, 1.0).to(torch.float32)
        sh = sh_from_rgb(rgb_fused)

        return cls(means=means, scales=scales, quats=quats, opacities=opacities, sh=sh)

    @classmethod
    def init_from_pointcloud_tensors(
        cls,
        pc_tensors: PointCloudTensors,
        device: device,
        *,
        voxel_size: VoxelTensor,
        base_scale: float,
        scale_mult: float = 1.0,
    ) -> Self:
        return cls.init_from_pointcloud(
            xyz=pc_tensors.xyz.to(device),
            rgb=pc_tensors.rgb.to(device),
            conf=pc_tensors.conf.to(device),
            voxel_size=voxel_size,
            base_scale=base_scale,
            scale_mult=scale_mult,
        )


def normalize_quat_tensor_(q: Float32[Tensor, "M 4"]) -> Float32[Tensor, "M 4"]:
    """ensure all quaternion magnitudes of 1. `_` suffix denotes in-place operations"""
    return q.div_(q.norm(dim=-1, keepdim=True).clamp_min(EPS))


def quats_identity(M: int, device: device) -> Float32[Tensor, "M 4"]:
    """creates tensor of quats where (x, y, z, w) = [0, 0, 0, 1]"""
    import torch

    q: Float32[Tensor, "M 4"] = torch.zeros((M, 4), device=device, dtype=torch.float32)
    q[:, 3] = 1.0
    return q


def w2c_3x4_to_view_4x4(extrinsic: TT.ExtrinsicTensor) -> Float32[Tensor, "B 4 4"]:
    """world-to-camera to view by simply adding homogeneous bottom row [0, 0, 0, 1]"""
    import torch

    B = extrinsic.shape[0]
    v: Float32[Tensor, "B 4 4"] = torch.zeros(
        (B, 4, 4), device=extrinsic.device, dtype=extrinsic.dtype
    )
    v[:, :3, :4] = extrinsic
    v[:, 3, 3] = 1.0
    return v


def sh_from_rgb(
    rgb: Union[Float32[Tensor, "M 3"], UInt8[Tensor, "M 3"]], sh_degree: int = 0
) -> Float32[Tensor, "M K 3"]:
    """derive spherical harmonics from RGB values, where only the DC (Direct Current) is filled in"""
    import torch

    device = rgb.device

    rgb_0_1: Float32[Tensor, "M 3"] = to_0_1(rgb)
    K = (sh_degree + 1) ** 2
    N = rgb_0_1.shape[0]

    sh: Float32[Tensor, "M K 3"] = torch.zeros(
        (N, K, 3), device=device, dtype=torch.float32
    )

    sh[:, 0, :] = rgb_0_1  # fill dimensions with RGB values, skipping `K` dimension

    return sh


def scales_from_confidence(
    conf: TT.ConfTensor,
    *,
    base_scale: float,
    scale_mult: float,
) -> Tensor:
    """use confidence values to initialize scale of splats"""
    import torch

    s = (float(base_scale) * float(scale_mult) / conf.clamp_min(1e-3)).to(torch.float32)
    return torch.stack([s, s, s], dim=1)


def scalars_from_xyz(
    xyz: Float32[Tensor, "N 3"],
    voxel_size_factor: float = 0.005,  # ~2.0m * 0.005 = 0.01 meters
    base_scale_factor: float = 0.02,  # corresponds to human height of ~2m
) -> Tuple[VoxelTensor, float]:
    """derive initial voxel size & base scale as factors of bounding box diagonal"""
    bbox = xyz.max(dim=0)[0] - xyz.min(dim=0)[0]
    diag = bbox.norm()  # `norm` returns a scalar tensor
    return diag * voxel_size_factor, (diag * base_scale_factor).item()


def fuse_points_by_voxel(
    xyz: TT.XYZTensor,
    rgb_0_1: TT.RGBTensor_0_1,
    conf: TT.ConfTensor,
    voxel_size: VoxelTensor,
) -> Tuple[
    Float32[Tensor, "M 3"],  # means
    Float32[Tensor, "M 3"],  # rgb_fused
    Float32[Tensor, "M"],
]:
    import torch

    assert rgb_0_1.dtype == torch.float32

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
    rgb_0_1 = rgb_0_1[order]
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
    means = (means / wt_sum[:, None]).to(torch.float32)

    # reduce rgb using a representative weighted AVERAGE rgb per voxel
    rgb_fused: Float32[Tensor, "M 3"] = torch.zeros(
        (M, 3), device=device, dtype=torch.float32
    )
    rgb_fused.scatter_add_(0, voxel_ids[:, None].expand(-1, 3), rgb_0_1 * wt[:, None])
    rgb_fused = rgb_fused / wt_sum[:, None]  # convert back to 'uint8'

    # reduce confidence using a representative MAX confidence per voxel
    conf_fused: Float32[Tensor, "M"] = (
        torch.zeros((M,), device=device, dtype=conf.dtype)
        .scatter_reduce_(0, voxel_ids, conf, reduce="amax")
        .to(torch.float32)
    )

    return means, rgb_fused, conf_fused


def save_ply_3dgs_binary(
    out_file: Path,
    model: SplatModel,
    encode_log_of_scale: bool = True,  # convert 'scale' as 'log(scale)'
    encode_opacity_in_logit: bool = True,  # convert 'opacity' 0-1 to unbounded logit space
) -> None:
    import numpy as np
    import torch

    assert model.means.shape[1] == 3
    assert model.scales.shape[1] == 3
    assert model.quats.shape[1] == 4
    assert model.opacities.shape[1] == 1

    means = model.means.detach().cpu().float()
    scales = model.scales.detach().cpu().float().clamp_min(EPS)
    quats = model.quats.detach().cpu().float()
    opacities = model.opacities.detach().cpu().float()

    if encode_log_of_scale:
        scales = scales.log()

    if encode_opacity_in_logit:
        x = opacities.clamp(EPS, 1.0 - EPS)
        opacities = torch.log(x) - torch.log1p(-x)

    sh = model.sh  # (M, K, 3)
    N, K, C = sh.shape

    # separate DC and rest
    f_dc = sh[:, 0, :]  # (M,3)
    f_rest = sh[:, 1:, :].reshape(N, -1)  # (M, 3*(K-1))

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
        f"element vertex {N}",
    ]
    header += [f"property {t} {n}" for (n, t) in props]
    header += ["end_header\n"]
    header_str = "\n".join(header)

    normals = torch.zeros((N, 3), dtype=torch.float32)  # set normals to zero

    body = torch.cat(
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
    ).numpy()

    with out_file.open(mode="wb") as f:
        f.write(header_str.encode("ascii"))
        f.write(body.astype(np.float32).tobytes())


def train_3dgs(
    model: SplatModel,
    images: Tensor,  # (B,3,H,W) unmasked
    masks: Tensor,  # (B,1,H,W) 1=valid, 0=ignore
    extrinsic: Tensor,
    intrinsic: Tensor,
    rasterizer: GsplatRasterizer,
    out_file: Path,
    config: TrainConfig = TrainConfig(),
) -> SplatModel:
    import torch
    from gsplat import DefaultStrategy

    device = images.device
    optimizers: Dict[str, torch.optim.Optimizer] = {
        name: torch.optim.Adam([param], lr=float(config.lr))
        for name, param in model.params.items()
    }

    strategy = DefaultStrategy()
    strategy.check_sanity(model.params, optimizers)
    strategy_state = strategy.initialize_state()

    B = images.shape[0]

    for step in range(int(config.steps)):
        idx = torch.randint(0, B, (config.cam_batch,), device=device)

        tgt = images.index_select(0, idx)
        msk = masks.index_select(0, idx)
        ex = extrinsic.index_select(0, idx)
        intr = intrinsic.index_select(0, idx)

        bg = torch.zeros(
            (config.cam_batch, 3, rasterizer.H, rasterizer.W),
            device=device,
            dtype=torch.float32,
        )

        pred, _, _, _ = rasterizer.render(model, ex, intr, bg)

        diff = (pred - tgt).abs()
        loss = (diff * msk).sum() / (msk.sum() + EPS)

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

        if step % 1000 == 0:
            with torch.no_grad():
                save_ply_3dgs_binary(out_file, model)

    return model


class GsplatRasterizer:
    def __init__(
        self,
        image_size: Tuple[int, int],
        sh_degree: int,
        znear: float = 0.01,
        zfar: float = 1000.0,
    ):
        self.H, self.W = image_size
        self.sh_degree = sh_degree
        self.znear = znear
        self.zfar = zfar

        from gsplat import rasterization

        self._rasterization = rasterization

    def render(
        self,
        model: SplatModel,
        extrinsic: Tensor,  # (B,3,4)
        intrinsic: Tensor,  # (B,3,3)
        bg: Tensor,  # (B,3,H,W)
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        import torch

        assert model.sh.shape[1] == (self.sh_degree + 1) ** 2

        device = model.means.device

        extrinsic = extrinsic.to(device=device, dtype=torch.float32)
        intrinsic = intrinsic.to(device=device, dtype=torch.float32)
        bg = bg.to(device=device, dtype=torch.float32)

        # convert (B,3,4) to (B,4,4)
        view = w2c_3x4_to_view_4x4(extrinsic)

        # gsplat expects opacities as (..., N)
        opacities = model.opacities.squeeze(-1)

        render_colors, render_alphas, meta = self._rasterization(
            means=model.means,  # (M,3)
            quats=model.quats,  # (M,4)
            scales=model.scales,  # (M,3)
            opacities=opacities,  # (M,)
            colors=model.sh,  # (M,K,3)
            sh_degree=self.sh_degree,
            viewmats=view,  # (B,4,4)
            Ks=intrinsic,  # (B,3,3)
            width=self.W,
            height=self.H,
            near_plane=self.znear,
            far_plane=self.zfar,
            backgrounds=bg.permute(0, 2, 3, 1),  # (B,H,W,3)
            render_mode="RGB+ED",
        )

        # render_colors: (B,H,W,4) to RGB + depth
        rgb = render_colors[..., :3]  # (B,H,W,3)
        depth = render_colors[..., 3:]  # (B,H,W,1)

        alpha = render_alphas  # (B,H,W,1)

        # radii stored in meta
        radii = meta["radii"]  # (num_intersections,)

        # convert to channel-first
        rgb = rgb.permute(0, 3, 1, 2)  # (B,3,H,W)
        depth = depth.permute(0, 3, 1, 2)  # (B,1,H,W)
        alpha = alpha.permute(0, 3, 1, 2)  # (B,1,H,W)

        return rgb, radii, depth, alpha
