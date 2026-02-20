from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

from ..infrastructure.macros import to_0_1
from ..infrastructure.schemas import PointCloudTensors

if TYPE_CHECKING:
    from torch import Tensor
    from torch import device as t_device


def _c2w_from_w2c(extrinsic: Tensor) -> Tensor:
    R = extrinsic[:, :3, :3]
    t = extrinsic[:, :3, 3]
    c = -(R.transpose(1, 2) @ t.unsqueeze(-1)).squeeze(-1)
    return c


def _projection_from_fov(
    tanfovx: Tensor,
    tanfovy: Tensor,
    znear: float,
    zfar: float,
) -> Tensor:
    import torch

    B = tanfovx.shape[0]
    P = torch.zeros((B, 4, 4), device=tanfovx.device, dtype=torch.float32)
    z_sign = 1.0

    P[:, 0, 0] = 1.0 / tanfovx
    P[:, 1, 1] = 1.0 / tanfovy
    P[:, 3, 2] = z_sign
    P[:, 2, 2] = z_sign * (zfar / (zfar - znear))
    P[:, 2, 3] = -(zfar * znear) / (zfar - znear)
    return P


@dataclass
class SplatParams:
    means: Tensor  # (M,3) float32
    scales: Tensor  # (M,3) float32 >0
    quats_xyzw: Tensor  # (M,4) float32 normalized
    opacity: Tensor  # (M,1) float32 [0,1] (or logits if you choose)
    colors: Tensor  # (M,3) float32 [0,1]
    sh: Tensor

    def to(self, device: t_device) -> SplatParams:
        return SplatParams(
            means=self.means.to(device),
            scales=self.scales.to(device),
            quats_xyzw=self.quats_xyzw.to(device),
            opacity=self.opacity.to(device),
            colors=self.colors.to(device),
            sh=self.colors.to(device),
        )


@dataclass
class TrainConfig:
    steps: int = 30_000
    lr: float = 1e-2
    cam_batch: int = 1
    l1_weight: float = 1.0
    clamp_scales_min: float = 1e-6
    renorm_quat_period: int = 1
    sh_degree: int = 0


def _norm_quat_xyzw(q: Tensor) -> Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + 1e-12)


def _quat_identity(M: int, device: t_device) -> Tensor:
    import torch

    q = torch.zeros((M, 4), device=device, dtype=torch.float32)
    q[:, 3] = 1.0
    return q


def _sh_from_rgb_u8(rgb_u8: Tensor, sh_degree: int = 0) -> Tensor:
    import torch

    rgb = to_0_1(rgb_u8)
    K = (sh_degree + 1) ** 2

    sh = torch.zeros((rgb.shape[0], K, 3), device=rgb.device, dtype=torch.float32)

    sh[:, 0, :] = rgb

    return sh


def init_scales_confidence(
    conf: Tensor,  # (N,)
    base_scale: float,
    scale_mult: float = 1.0,
) -> Tensor:
    import torch

    s = (float(base_scale) * float(scale_mult) / conf.clamp_min(1e-3)).to(torch.float32)
    return torch.stack([s, s, s], dim=1)


def _w2c_3x4_to_view_4x4(extrinsic: Tensor) -> Tensor:
    """world-to-camera 3x4 to view 4x4"""

    import torch

    B = extrinsic.shape[0]
    v = torch.zeros((B, 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
    v[:, :3, :4] = extrinsic
    v[:, 3, 3] = 1.0
    return v


def fuse_points_voxel_hash(
    xyz: Tensor,  # (N,3) float32
    rgb_u8: Tensor,  # (N,3) uint8
    conf: Tensor,  # (N,) float32
    voxel: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    import torch

    if xyz.numel() == 0:
        device = xyz.device
        return (
            torch.empty((0, 3), device=device, dtype=torch.float32),
            torch.empty((0, 3), device=device, dtype=torch.uint8),
            torch.empty((0,), device=device, dtype=torch.float32),
        )

    device = xyz.device
    v = torch.tensor(voxel, device=device, dtype=xyz.dtype)
    key = torch.floor(xyz / v).to(torch.int64)

    hx = key[:, 0] * 73856093
    hy = key[:, 1] * 19349663
    hz = key[:, 2] * 83492791
    h = (hx ^ hy ^ hz).to(torch.int64)

    order = torch.argsort(h)
    h = h[order]
    xyz = xyz[order]
    rgb_u8 = rgb_u8[order]
    conf = conf[order]

    new_group = torch.ones_like(h, dtype=torch.bool)
    new_group[1:] = h[1:] != h[:-1]
    group_id = torch.cumsum(new_group.to(torch.int64), dim=0) - 1
    M = int(group_id[-1].item()) + 1

    w = conf.clamp_min(1e-6).to(xyz.dtype)
    wsum = torch.zeros((M,), device=device, dtype=xyz.dtype).scatter_add_(
        0, group_id, w
    )

    means = torch.zeros((M, 3), device=device, dtype=xyz.dtype)
    means.scatter_add_(0, group_id[:, None].expand(-1, 3), xyz * w[:, None])
    means = means / wsum[:, None]

    rgb = rgb_u8.to(torch.float32)
    rgb_acc = torch.zeros((M, 3), device=device, dtype=torch.float32)
    rgb_acc.scatter_add_(
        0, group_id[:, None].expand(-1, 3), rgb * w[:, None].to(torch.float32)
    )
    rgb_acc = (
        (rgb_acc / wsum[:, None].to(torch.float32)).clamp(0.0, 255.0).to(torch.uint8)
    )

    conf_out = torch.zeros((M,), device=device, dtype=conf.dtype).scatter_reduce_(
        0, group_id, conf, reduce="amax"
    )
    return means.to(torch.float32), rgb_acc, conf_out.to(torch.float32)


def estimate_tangent_frame_from_pointmap(
    pointmap: Tensor,  # (B,H,W,3)
    conf_map: Tensor,  # (B,H,W)
) -> Tuple[Tensor, Tensor]:
    import torch
    import torch.nn.functional as F

    dx = pointmap[:, :, 2:, :] - pointmap[:, :, :-2, :]
    dy = pointmap[:, 2:, :, :] - pointmap[:, :-2, :, :]

    dx = F.pad(dx, (0, 0, 1, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 0, 1, 1))

    n = torch.cross(dx, dy, dim=-1)
    n = n / (n.norm(dim=-1, keepdim=True) + 1e-12)

    t = dx / (dx.norm(dim=-1, keepdim=True) + 1e-12)
    b = torch.cross(n, t, dim=-1)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-12)

    m = conf_map > 1e-5
    z = torch.zeros_like(n)
    n = torch.where(m[..., None], n, z)
    t = torch.where(m[..., None], t, z)
    b = torch.where(m[..., None], b, z)

    frame = torch.stack([t, b, n], dim=-2)  # (B,H,W,3,3)
    return frame, m


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

    scales = init_scales_confidence(
        conf_fused, base_scale=base_scale, scale_mult=scale_mult
    ).to(device)
    quats = _norm_quat_xyzw(_quat_identity(M, device))
    opacity = conf_fused.clamp(0.0, 1.0).unsqueeze(1).to(torch.float32)
    colors = to_0_1(rgb_fused).to(device)
    sh = _sh_from_rgb_u8(rgb_0_255)

    return SplatParams(
        means=means,
        scales=scales,
        quats_xyzw=quats,
        opacity=opacity,
        colors=colors,
        sh=sh,
    )


def train_3dgs(
    params: SplatParams,
    images: Tensor,  # (B,3,H,W) float32 [0,1]
    extrinsic: Tensor,  # (B,3,4)
    intrinsic: Tensor,  # (B,3,3)
    rasterizer: GsplatRasterizer,
    cfg: TrainConfig = TrainConfig(),
) -> SplatParams:
    import torch

    device = images.device
    params = params.to(device)

    params.means.requires_grad_(True)
    params.scales.requires_grad_(True)
    params.quats_xyzw.requires_grad_(True)
    params.opacity.requires_grad_(True)
    params.colors.requires_grad_(True)
    params.sh.requires_grad_(True)

    opt = torch.optim.Adam(
        [
            params.means,
            params.scales,
            params.quats_xyzw,
            params.opacity,
            params.colors,
            params.sh,
        ],
        lr=float(cfg.lr),
    )

    B = images.shape[0]
    bg = torch.zeros(
        (cfg.cam_batch, 3, rasterizer.H, rasterizer.W),
        device=device,
        dtype=torch.float32,
    )

    for iter in range(int(cfg.steps)):
        idx = torch.randint(0, B, (cfg.cam_batch,), device=device)
        tgt = images.index_select(0, idx)
        ex = extrinsic.index_select(0, idx)
        intr = intrinsic.index_select(0, idx)

        pred, _, _, _ = rasterizer.render(params, ex, intr, bg)
        loss = float(cfg.l1_weight) * (pred - tgt).abs().mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if cfg.renorm_quat_period > 0 and (iter % cfg.renorm_quat_period) == 0:
            with torch.no_grad():
                params.quats_xyzw[:] = _norm_quat_xyzw(params.quats_xyzw)
                params.scales[:] = params.scales.clamp_min(float(cfg.clamp_scales_min))

    return params


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
    assert params.opacity.shape[1] == 1
    assert params.colors.shape[1] == 3

    means = params.means.detach().cpu().float()
    scales = params.scales.detach().cpu().float().clamp_min(1e-12)
    quats = params.quats_xyzw.detach().cpu().float()
    opacity = params.opacity.detach().cpu().float()
    colors = params.colors.detach().cpu().float()  # (M,3)

    if encode_scale_log:
        scales = scales.log()

    if encode_opacity_logit:
        x = opacity.clamp(1e-6, 1.0 - 1e-6)
        opacity = torch.log(x) - torch.log1p(-x)

    M = means.shape[0]

    # Degree-0 only
    f_dc = colors
    f_rest = torch.zeros((M, 0), dtype=torch.float32)

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
            opacity,
            scales,
            quats,
        ],
        dim=1,
    ).numpy()

    with open(out_file, "wb") as f:
        f.write(header_s.encode("ascii"))
        f.write(row.astype(np.float32).tobytes())


def init_from_pointcloud_tensors(
    pct: PointCloudTensors,
    voxel: float,
    base_scale: float,
    scale_mult: float = 1.0,
    device: Optional[t_device] = None,
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

        device = params.means.device

        extrinsic = extrinsic.to(device=device, dtype=torch.float32)
        intrinsic = intrinsic.to(device=device, dtype=torch.float32)
        bg = bg.to(device=device, dtype=torch.float32)

        # convert (B,3,4) to (B,4,4)
        view = _w2c_3x4_to_view_4x4(extrinsic)

        # gsplat expects opacities as (..., N)
        opacities = params.opacity.squeeze(-1)

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
