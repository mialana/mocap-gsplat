"""
`M` corresponds to # of splats
`N` corresponds to # of reconstructed points
`K` corresponds to # of SH coefficients per splat. `K = (sh_degree+1)^2)`
`S` corresponds to scene size, which would be # of cameras capturing each frame

try to keep as lazy import as there is top-level import of `dltype` and `torch`
"""

from __future__ import annotations

from multiprocessing import Queue
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated as Anno,
    Dict,
    Literal,
    NamedTuple,
    Self,
    Tuple,
    TypeAlias,
)

from ..infrastructure.dl_ops import (
    PointCloudTensors,
    TensorTypes as TT,
    UInt8Float32Tensor,
    to_0_1,
    to_channel_as_primary,
)
from ..infrastructure.macros import add_suffix_to_path
from ..infrastructure.schemas import (
    SplatTrainingConfig,
    SplatTrainingStats,
    UnexpectedError,
)

HASH_MULTIPLIER_X = 73856093
HASH_MULTIPLIER_Y = 19349663
HASH_MULTIPLIER_Z = 83492791

SH_C0 = 0.28209479177387814

EPS: float = 1e-6
INF: float = float(1e10)  # effectively infinity for our purposes

import dltype
import torch

if TYPE_CHECKING:
    TrainingQueueType: TypeAlias = Queue[Tuple[str, str]]
else:
    TrainingQueueType: TypeAlias = Queue


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
def w2c_3x4_to_viewmats_4x4(
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
    rgb: Anno[torch.Tensor, UInt8Float32Tensor["M 3"]], sh_degree: int
) -> Anno[torch.Tensor, dltype.Float32Tensor["M K 3"]]:
    """derive spherical harmonics from RGB values, where only the DC (Direct Current) is filled in"""

    device = rgb.device

    rgb_0_1 = to_0_1(rgb)

    K = (sh_degree + 1) ** 2
    N = rgb.shape[0]

    sh: Anno[torch.Tensor, dltype.Float32Tensor["M K 3"]] = torch.zeros(
        (N, K, 3), device=device, dtype=torch.float32
    )

    # fill dimensions with RGB values, skipping `K` dimension
    sh[:, 0, :] = (rgb_0_1 - 0.5) / SH_C0  # zero-center for stable training
    return sh


@dltype.dltyped()
def scales_from_knn_means(
    means: Anno[torch.Tensor, dltype.Float32Tensor["M 3"]],
    neighborhood_size: int,
    multiplier: float,
) -> Anno[torch.Tensor, dltype.Float32Tensor["M 3"]]:
    """use k nearest neighbors means to initialize scale of splats"""
    # pre-specify tensor types. `NS` = neighborhood_size
    pairs: Anno[torch.Tensor, dltype.Float32Tensor["M M"]]
    knn_dists: Anno[torch.Tensor, dltype.Float32Tensor["M NS"]]
    dists2_avg: Anno[torch.Tensor, dltype.Float32Tensor["M"]]  # ~= dists_avg, scales

    # compute pairwise distances
    pairs = torch.cdist(means, means)

    # ignore distances to self by adding a large value to diagonals
    pairs.fill_diagonal_(float(INF))

    # get k nearest neighbors.
    knn_dists, _ = torch.topk(pairs, k=neighborhood_size, largest=False)

    # get means of squared `knn_dists`
    dists2_avg = (knn_dists**2).mean(dim=-1)
    dists_avg = dists2_avg.sqrt_()

    # scales should be in log space
    scales = torch.log_(dists_avg * multiplier)

    return torch.stack([scales, scales, scales], dim=1)


@dltype.dltyped()
def opacities_from_confidence(
    conf: Anno[torch.Tensor, dltype.Float32Tensor["M"]],
) -> Anno[torch.Tensor, dltype.Float32Tensor["M"]]:
    return conf.clamp(EPS, 1.0 - EPS).logit_(EPS)


@dltype.dltyped()
def voxel_size_from_means(
    means: Anno[torch.Tensor, dltype.Float32Tensor["N 3"]],
    voxel_size_factor: float,
) -> TT.VoxelTensor:
    """derive initial voxel size as a factor of bounding box diagonal"""
    bbox = means.max(dim=0)[0] - means.min(dim=0)[0]
    diag = bbox.norm()  # `norm` returns a scalar tensor
    return diag * voxel_size_factor


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
    """cull points to a representative per voxel"""

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
        scales: Anno[torch.Tensor, dltype.Float32Tensor["M 3"]],  # log space
        quats: Anno[torch.Tensor, dltype.Float32Tensor["M 4"]],
        opacities: Anno[torch.Tensor, dltype.Float32Tensor["M"]],  # stored logit space
        sh: Anno[torch.Tensor, dltype.Float32Tensor["M K 3"]],
    ):

        super().__init__()

        # store spherical harmonics as separate parameters
        sh0 = sh[:, :1, :]
        shN = sh[:, 1:, :]

        self.params: torch.nn.ParameterDict = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(means),
                "scales": torch.nn.Parameter(scales),
                "quats": torch.nn.Parameter(quats),
                "opacities": torch.nn.Parameter(opacities),
                "sh0": torch.nn.Parameter(sh0),
                "shN": torch.nn.Parameter(shN),
            }
        )
        self.to(device)  # this moves all parameters to the device

    @property
    def means_(self) -> Anno[torch.Tensor, dltype.Float32Tensor["M 3"]]:
        return self.params.get_parameter("means")

    @property
    def scales_(self) -> Anno[torch.Tensor, dltype.Float32Tensor["M 3"]]:
        return self.params.get_parameter("scales")

    @property
    def quats_(self) -> Anno[torch.Tensor, dltype.Float32Tensor["M 4"]]:
        return self.params.get_parameter("quats")

    @property
    def opacities_(self) -> Anno[torch.Tensor, dltype.Float32Tensor["M"]]:
        return self.params.get_parameter("opacities")

    @property
    def sh0_(self) -> Anno[torch.Tensor, dltype.Float32Tensor["M 1 3"]]:
        return self.params.get_parameter("sh0")

    @property
    def shN_(self) -> Anno[torch.Tensor, dltype.Float32Tensor["M Kminus1 3"]]:
        return self.params.get_parameter("shN")

    @torch.no_grad
    def post_step(self):
        # q = self.quats_
        # normalize_quat_tensor_(q)
        pass  # rasterization does normalization internally

    @property
    def detached_tensors(self) -> ModelTensors:
        return self.ModelTensors(
            **{
                name: param.detach().clone()
                for name, param in self.params.named_parameters()
            }
        )

    @classmethod
    def init_from_pointcloud_tensors(
        cls,
        pct: PointCloudTensors,
        device: torch.device,
        *,
        neighborhood_size: int = 3,  # for `scales` initialization
        scales_multiplier: float = 1.5,
        sh_degree: int = 0,
        ###
        fuse_by_voxel: bool = False,
        voxel_size_factor: float = 0.005,  # ~2.0m person * 0.005 = 0.01 meters
        ###
        init_tactics: Literal["custom", "gsplat"] = "custom",
        opacity_initial: float = 0.1,  # if `init_tactics` is `gsplat`
    ) -> Self:
        pct.to(device)  # convert all tensors to device

        means = pct.xyz
        rgb = to_0_1(pct.rgb_0_255)
        conf = pct.conf

        if fuse_by_voxel:
            voxel_size = voxel_size_from_means(means, voxel_size_factor)
            means, rgb, conf = fuse_points_by_voxel(means, rgb, conf, voxel_size)

        M = means.shape[0]

        scales = scales_from_knn_means(means, neighborhood_size, scales_multiplier)

        if init_tactics == "gsplat":
            quats = torch.rand((M, 4), device=device)
            opacities = torch.full((M,), opacity_initial, device=device).logit_()
        else:
            quats = quats_identity(M, device)
            opacities = opacities_from_confidence(conf)

        sh = sh_from_rgb(rgb, sh_degree)

        return cls(
            device, means=means, scales=scales, quats=quats, opacities=opacities, sh=sh
        )

    @dltype.dltyped_namedtuple()
    class ModelTensors(NamedTuple):
        means: Anno[torch.Tensor, dltype.Float32Tensor["M 3"]]
        scales: Anno[torch.Tensor, dltype.Float32Tensor["M 3"]]
        quats: Anno[torch.Tensor, dltype.Float32Tensor["M 4"]]
        opacities: Anno[torch.Tensor, dltype.Float32Tensor["M"]]
        sh0: Anno[torch.Tensor, dltype.Float32Tensor["M 1 3"]]
        shN: Anno[torch.Tensor, dltype.Float32Tensor["M Kminus1 3"]]


class GsplatRasterizer:
    """
    utilizes `gsplat.rasterization` method internally, with custom `SplatModel` module.
    """

    def __init__(
        self,
        device: torch.device,
        image_HW: Tuple[int, int],
        *,
        sh_degree: int = 0,
        znear: float = 0.01,
        zfar: float = 1000.0,
        absgrad: bool = True,
    ):
        self.H, self.W = image_HW
        self.device = device
        self.sh_degree = sh_degree
        self.znear = znear
        self.zfar = zfar
        self.absgrad = absgrad
        self.packed = True  # there is a bug in the rasterization pipeline so this should always be `True`

    @dltype.dltyped()
    def render(
        self,
        model: SplatModel,
        viewmats: Anno[torch.Tensor, dltype.Float32Tensor["S 4 4"]],
        intrinsic: TT.IntrinsicTensor,
    ) -> Tuple[
        Anno[torch.Tensor, dltype.Float32Tensor["S 3 H W"]],
        Anno[torch.Tensor, dltype.Float32Tensor["S 1 H W"]],
        Anno[torch.Tensor, dltype.Float32Tensor["S 1 H W"]],
        Dict,
    ]:
        from gsplat import rasterization

        means = model.means_
        quats = model.quats_  # rasterization does normalization internally
        scales = model.scales_.exp()  # was stored in log space
        opacities = model.opacities_.sigmoid()  # was stored in logit space

        colors = torch.cat([model.sh0_, model.shN_], dim=1)

        assert colors.shape[1] == (self.sh_degree + 1) ** 2

        viewmats = viewmats.to(self.device)
        intrinsic = intrinsic.to(self.device)

        render_colors, render_alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=intrinsic,
            width=self.W,
            height=self.H,
            near_plane=self.znear,
            far_plane=self.zfar,
            sh_degree=self.sh_degree,
            packed=self.packed,
            render_mode="RGB+ED",
            absgrad=self.absgrad,
            rasterize_mode="antialiased",
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


def save_ply_3dgs(out_file: Path, tensors: SplatModel.ModelTensors):
    import numpy as np

    means, scales, quats, opacities, sh0, shN = tensors

    M = means.shape[0]
    # opacities can use view as the data is contiguous in memory
    opacities: Anno[torch.Tensor, dltype.Float32Tensor["M 1"]] = opacities.view(M, 1)
    sh0: Anno[torch.Tensor, dltype.Float32Tensor["M 3"]] = sh0.reshape(M, -1)
    shN: Anno[torch.Tensor, dltype.Float32Tensor["M 3xKminus1"]] = shN.reshape(M, -1)

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

    num_rest = shN.shape[1]
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
        f"format binary_little_endian 1.0",
        f"element vertex {M}",
    ]
    header += [f"property {t} {n}" for (n, t) in props]
    header += ["end_header\n"]
    header_str = "\n".join(header)

    normals = torch.zeros(
        (M, 3), dtype=torch.float32, device=means.device
    )  # set normals to zero

    body = (
        torch.cat([means, normals, sh0, shN, opacities, scales, quats], dim=1)
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
    queue: TrainingQueueType,
    *,
    verbose: bool = True,
) -> SplatModel:
    from gsplat import DefaultStrategy

    device = rasterizer.device
    absgrad = rasterizer.absgrad
    packed = rasterizer.packed

    images_0_1 = to_0_1(images)
    images_alpha_0_1 = to_0_1(images_alpha)

    pct.to(device)
    viewmats = w2c_3x4_to_viewmats_4x4(pct.extrinsic)
    intrinsic = pct.intrinsic

    mask = images_alpha_0_1 > 0.5
    bg_mask = ~mask

    # use different learning rates per parameter
    optimizers: Dict[str, torch.optim.Optimizer] = {
        name: torch.optim.Adam([param], lr=config.lr[idx])
        for idx, (name, param) in enumerate(model.params.named_parameters())
    }  # gsplat expects one optimizer per parameter

    strategy = DefaultStrategy(
        absgrad=absgrad,
        verbose=verbose,
        grow_grad2d=config.refine_grow_threshold,
        refine_start_iter=(
            int(INF) if config.refine_start_step == -1 else config.refine_start_step
        ),
        refine_stop_iter=config.refine_end_step,
        refine_every=config.refine_interval,
        reset_every=(
            int(INF)
            if config.reset_opacity_interval == -1
            else config.reset_opacity_interval
        ),
        revised_opacity=config.revised_opacities_heuristic,
    )
    strategy.check_sanity(model.params, optimizers)
    strategy_state = strategy.initialize_state()

    S: int = images_0_1.shape[0]

    msg = f"Beginning training with '{model.means_.shape[0]}' initial splats."
    queue.put(("update", msg))

    opacity_loss_item = None
    for step in range(config.steps):
        if model.means_.shape[0] == 0:
            raise UnexpectedError("All splats were pruned. Stopping training.")

        idx = torch.randint(0, S, (config.scene_size,), device=device)

        rgb = images_0_1.index_select(0, idx)
        view = viewmats.index_select(0, idx)
        intri = intrinsic.index_select(0, idx)

        pred_rgb, pred_depth, pred_alpha, meta = rasterizer.render(model, view, intri)

        msk = mask.index_select(0, idx)
        bg_msk = bg_mask.index_select(0, idx)

        msk_rgb = msk.expand(-1, 3, -1, -1)
        rgb_loss = (pred_rgb - rgb)[msk_rgb].abs().mean()

        loss = rgb_loss  # TODO: implement PSNR loss and eval method

        # penalize alpha values within background
        alpha_loss = pred_alpha[bg_msk].mean()
        loss += alpha_loss * config.alpha_lambda

        if not config.revised_opacities_heuristic:
            # penalize opacities for being far from either 0 or 1
            opa_normalized = model.opacities_.sigmoid()
            opacity_loss = (opa_normalized.mul(torch.sub(1.0, opa_normalized))).mean()
            opacity_loss_item = opacity_loss.item()

            loss += opacity_loss * config.opacity_lambda

        strategy.step_pre_backward(model.params, optimizers, strategy_state, step, meta)

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)

        loss.backward()

        strategy.step_post_backward(
            model.params, optimizers, strategy_state, step, meta, packed=packed
        )

        for opt in optimizers.values():
            opt.step()

        if step % config.save_ply_interval == 0 or step == config.steps - 1:
            torch.cuda.synchronize()
            with torch.no_grad():
                out_file = (
                    add_suffix_to_path(ply_out_file, f".{step:06d}")
                    if config.increment_ply_file
                    else ply_out_file
                )
                save_ply_3dgs(out_file, model.detached_tensors)

        if step % 1000 == 0 or step == config.steps - 1:
            stats = SplatTrainingStats(
                step=step,
                num_splats=model.means_.shape[0],
                rgb_loss=rgb_loss.item(),
                alpha_loss=alpha_loss.item(),
                opacity_loss=opacity_loss_item,
                total_loss=loss.item(),
            )
            queue.put(("update", f"Current Training Stats:\n{str(stats)}"))
    return model
