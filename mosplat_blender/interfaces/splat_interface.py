from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from ..infrastructure.macros import to_0_1
from ..infrastructure.schemas import PointCloudTensors

if TYPE_CHECKING:
    from torch import Tensor, device
    from torch.nn import ParameterDict


@dataclass
class SplatParams:
    means: Tensor  # (M,3) float32
    scales: Tensor  # (M,3) float32 >0
    quats_xyzw: Tensor  # (M,4) float32 normalized
    opacities: Tensor
    sh: Tensor

    def to(self, device: device) -> SplatParams:
        return SplatParams(
            means=self.means.to(device),
            scales=self.scales.to(device),
            quats_xyzw=self.quats_xyzw.to(device),
            opacities=self.opacities.to(device),
            sh=self.sh.to(device),
        )

    def to_parameter_dict(self, device: device) -> ParameterDict:
        import torch

        return torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(self.means.to(device)),
                "scales": torch.nn.Parameter(self.scales.to(device)),
                "quats": torch.nn.Parameter(self.quats_xyzw.to(device)),
                "opacities": torch.nn.Parameter(self.opacities.to(device)),
                "sh": torch.nn.Parameter(self.sh.to(device)),
            }
        )


@dataclass
class TrainConfig:
    steps: int = 30_000
    lr: float = 1e-2
    sh_degree: int = 0
    cam_batch: int = 1
    l1_weight: float = 1.0
    clamp_scales_min: float = 1e-6
    renorm_quat_period: int = 1


def norm_quat_xyzw(q: Tensor) -> Tensor:
    """ensure quaternion magnitude of 1"""
    return q / (q.norm(dim=-1, keepdim=True) + 1e-12)


def quat_identity(M: int, device: device) -> Tensor:
    """creates a quat where (x, y, z, w) = [0, 0, 0, 1]"""
    import torch

    q = torch.zeros((M, 4), device=device, dtype=torch.float32)
    q[:, 3] = 1.0
    return q


def voxel_size_from_xyz(xyz: Tensor, factor: float = 0.005):
    """derive voxel size as a reasonable factor of bounding box diagonal"""
    bbox = xyz.max(dim=0)[0] - xyz.min(dim=0)[0]
    diag = bbox.norm()
    return diag * factor


def sh_from_rgb_0_255(rgb_0_255: Tensor, sh_degree: int = 0) -> Tensor:
    """derive spherical harmonics from RGB values, where only the DC (Direct Current) is filled in"""
    import torch

    rgb_0_1 = to_0_1(rgb_0_255)
    K = (sh_degree + 1) ** 2

    sh = torch.zeros(
        (rgb_0_1.shape[0], K, 3), device=rgb_0_1.device, dtype=torch.float32
    )

    sh[:, 0, :] = rgb_0_1  # fill dimensions with RGB values, skipping `K` dimension

    return sh


def init_scales_from_confidence(
    conf: Tensor,  # (N,)
    base_scale: float,
    scale_mult: float = 1.0,
) -> Tensor:
    """use confidence values to initialize scale of splats"""
    import torch

    s = (float(base_scale) * float(scale_mult) / conf.clamp_min(1e-3)).to(torch.float32)
    return torch.stack([s, s, s], dim=1)


def _w2c_3x4_to_view_4x4(extrinsic: Tensor) -> Tensor:
    """world-to-camera to view by simply adding homogeneous bottom row [0, 0, 0, 1]"""
    import torch

    B = extrinsic.shape[0]
    v = torch.zeros((B, 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
    v[:, :3, :4] = extrinsic
    v[:, 3, 3] = 1.0
    return v


def fuse_points_voxel_hash(
    xyz: Tensor,  # (N,3) float32
    rgb_0_255: Tensor,  # (N,3) uint8
    conf: Tensor,  # (N,) float32
    voxel: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    import torch

    if xyz.numel() == 0:  # `numel` is num elements
        device = xyz.device
        return (
            torch.empty((0, 3), device=device, dtype=torch.float32),
            torch.empty((0, 3), device=device, dtype=torch.uint8),
            torch.empty((0,), device=device, dtype=torch.float32),
        )

    device = xyz.device
    v = torch.tensor(voxel, device=device, dtype=xyz.dtype)
    # map continuous 3D points to integer voxel grid
    key = torch.floor(xyz / v).to(torch.int64)

    hx = key[:, 0] * 73856093
    hy = key[:, 1] * 19349663
    hz = key[:, 2] * 83492791
    h = (hx ^ hy ^ hz).to(torch.int64)  # creates a 1D hash key

    order = torch.argsort(h)  # define the order using sorted hash key
    # apply order, effectively sorting by hash so that equal voxels become contiguous
    h = h[order]
    xyz = xyz[order]
    rgb_0_255 = rgb_0_255[order]
    conf = conf[order]

    new_group = torch.ones_like(h, dtype=torch.bool)  # `True`'s are group boundaries
    new_group[1:] = h[1:] != h[:-1]  # create groups of contiguous equal voxels
    group_id = torch.cumsum(new_group.to(torch.int64), dim=0) - 1  # id to each group
    M: int = int(group_id[-1].item()) + 1

    wt = conf.clamp_min(1e-6).to(xyz.dtype)  # weight from confidence
    wt_sum = torch.zeros((M,), device=device, dtype=xyz.dtype).scatter_add_(
        0, group_id, wt
    )

    # get weighted average positions based off `xyz`
    means = torch.zeros((M, 3), device=device, dtype=xyz.dtype)
    means.scatter_add_(0, group_id[:, None].expand(-1, 3), xyz * wt[:, None])
    means = (means / wt_sum[:, None]).to(torch.float32)

    # get weighted average rgb
    rgb = rgb_0_255.to(torch.float32)
    rgb_acc = torch.zeros((M, 3), device=device, dtype=torch.float32)
    rgb_acc.scatter_add_(
        0, group_id[:, None].expand(-1, 3), rgb * wt[:, None].to(torch.float32)
    )
    rgb_acc = (
        (rgb_acc / wt_sum[:, None].to(torch.float32)).clamp(0.0, 255.0).to(torch.uint8)
    )  # convert back to 'uint8'

    # set voxel confidence based on the max confidence of views of voxel
    conf_out = (
        torch.zeros((M,), device=device, dtype=conf.dtype)
        .scatter_reduce_(0, group_id, conf, reduce="amax")
        .to(torch.float32)
    )
    return means, rgb_acc, conf_out


def init_splats_from_pointcloud(
    xyz: Tensor,  # (N,3) float32
    rgb_0_255: Tensor,  # (N,3) uint8
    conf: Tensor,  # (N,) float32
    voxel: float,
    base_scale: float,
    scale_mult: float = 1.0,
) -> SplatParams:
    import torch

    means, rgb_fused, conf_fused = fuse_points_voxel_hash(
        xyz, rgb_0_255, conf, voxel=voxel
    )
    M = means.shape[0]
    device = means.device

    scales = init_scales_from_confidence(
        conf_fused, base_scale=base_scale, scale_mult=scale_mult
    ).to(device)
    quats = norm_quat_xyzw(quat_identity(M, device))
    opacity = conf_fused.clamp(0.0, 1.0).to(torch.float32)
    sh = sh_from_rgb_0_255(rgb_fused)

    return SplatParams(
        means=means,
        scales=scales,
        quats_xyzw=quats,
        opacities=opacity,
        sh=sh,
    )


def init_from_pointcloud_tensors(
    pct: PointCloudTensors,
    voxel: float,
    base_scale: float,
    scale_mult: float = 1.0,
    device: Optional[device] = None,
) -> SplatParams:
    dev = device if device is not None else pct.xyz.device
    return init_splats_from_pointcloud(
        xyz=pct.xyz.to(dev),
        rgb_0_255=pct.rgb.to(dev),
        conf=pct.conf.to(dev),
        voxel=voxel,
        base_scale=base_scale,
        scale_mult=scale_mult,
    )


def save_ply_3dgs_binary(
    out_file: Path,
    params: SplatParams,
    encode_scale_log: bool = True,
    encode_opacity_logit: bool = True,
) -> None:
    import numpy as np
    import torch

    assert params.means.shape[1] == 3
    assert params.scales.shape[1] == 3
    assert params.quats_xyzw.shape[1] == 4
    assert params.opacities.shape[1] == 1

    means = params.means.detach().cpu().float()
    scales = params.scales.detach().cpu().float().clamp_min(1e-12)
    quats = params.quats_xyzw.detach().cpu().float()
    opacity = params.opacities.detach().cpu().float()

    if encode_scale_log:
        scales = scales.log()

    if encode_opacity_logit:
        x = opacity.clamp(1e-6, 1.0 - 1e-6)
        opacity = torch.log(x) - torch.log1p(-x)

    sh = params.sh  # (M, K, 3)
    M, K, _ = sh.shape

    # separate DC and rest
    f_dc = sh[:, 0, :]  # (M,3)
    f_rest = sh[:, 1:, :].reshape(M, -1)  # (M, 3*(K-1))

    props = (
        ("x", "float"),
        ("y", "float"),
        ("z", "float"),
        ("nx", "float"),
        ("ny", "float"),
        ("nz", "float"),
        ("f_dc_0", "float"),
        ("f_dc_1", "float"),
        ("f_dc_2", "float"),
        ("opacity", "float"),
        ("scale_0", "float"),
        ("scale_1", "float"),
        ("scale_2", "float"),
        ("rot_0", "float"),
        ("rot_1", "float"),
        ("rot_2", "float"),
        ("rot_3", "float"),
    )

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {M}",
    ]
    header += [f"property {t} {n}" for (n, t) in props]
    header += ["end_header\n"]
    header_s = "\n".join(header)

    zeros_n = torch.zeros((M, 3), dtype=torch.float32)

    row = torch.cat(
        [
            means,
            zeros_n,
            f_dc,
            f_rest,
            opacity,
            scales,
            quats,
        ],
        dim=1,
    ).numpy()

    with open(out_file, "wb") as f:
        f.write(header_s.encode("ascii"))
        f.write(row.astype(np.float32).tobytes())


def train_3dgs(
    params: SplatParams,
    images: Tensor,
    extrinsic: Tensor,
    intrinsic: Tensor,
    rasterizer: GsplatRasterizer,
    config: TrainConfig = TrainConfig(),
) -> SplatParams:
    import torch
    from gsplat import DefaultStrategy

    device = images.device

    param_dict = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(params.means.to(device)),
            "scales": torch.nn.Parameter(params.scales.to(device)),
            "quats": torch.nn.Parameter(params.quats_xyzw.to(device)),
            "opacities": torch.nn.Parameter(params.opacities.squeeze(-1).to(device)),
            "sh": torch.nn.Parameter(params.sh.to(device)),
        }
    )

    optimizers: Dict[str, torch.optim.Optimizer] = {
        k: torch.optim.Adam([v], lr=float(config.lr)) for k, v in param_dict.items()
    }

    strategy = DefaultStrategy()
    strategy.check_sanity(param_dict, optimizers)
    strategy_state = strategy.initialize_state()

    B = images.shape[0]

    for step in range(int(config.steps)):
        idx = torch.randint(0, B, (config.cam_batch,), device=device)

        tgt = images.index_select(0, idx)
        ex = extrinsic.index_select(0, idx)
        intr = intrinsic.index_select(0, idx)

        bg = torch.zeros(
            (config.cam_batch, 3, rasterizer.H, rasterizer.W),
            device=device,
            dtype=torch.float32,
        )

        splat_params = SplatParams(
            means=param_dict["means"],
            scales=param_dict["scales"],
            quats_xyzw=param_dict["quats"],
            opacities=param_dict["opacities"].unsqueeze(-1),
            sh=param_dict["sh"],
        )

        pred, _, _, _ = rasterizer.render(splat_params, ex, intr, bg)

        loss = float(config.l1_weight) * (pred - tgt).abs().mean()

        strategy.step_pre_backward(
            param_dict,
            optimizers,
            strategy_state,
            step,
            {},
        )

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)

        loss.backward()

        strategy.step_post_backward(
            param_dict,
            optimizers,
            strategy_state,
            step,
            {},
        )

        for opt in optimizers.values():
            opt.step()

    return SplatParams(
        means=param_dict["means"].detach(),
        scales=param_dict["scales"].detach(),
        quats_xyzw=norm_quat_xyzw(param_dict["quats"].detach()),
        opacities=param_dict["opacities"].detach().unsqueeze(1),
        sh=param_dict["sh"].detach(),
    )


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
        params: SplatParams,
        extrinsic: Tensor,  # (B,3,4)
        intrinsic: Tensor,  # (B,3,3)
        bg: Tensor,  # (B,3,H,W)
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        import torch

        assert params.sh.shape[1] == (self.sh_degree + 1) ** 2

        device = params.means.device

        extrinsic = extrinsic.to(device=device, dtype=torch.float32)
        intrinsic = intrinsic.to(device=device, dtype=torch.float32)
        bg = bg.to(device=device, dtype=torch.float32)

        # convert (B,3,4) to (B,4,4)
        view = _w2c_3x4_to_view_4x4(extrinsic)

        # gsplat expects opacities as (..., N)
        opacities = params.opacities.squeeze(-1)

        render_colors, render_alphas, meta = self._rasterization(
            means=params.means,  # (M,3)
            quats=params.quats_xyzw,  # (M,4)
            scales=params.scales,  # (M,3)
            opacities=opacities,  # (M,)
            colors=params.sh,  # (M,K,3)
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
