"""
provides the interface between the add-on and the VGGT model.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import multiprocessing.synchronize as mp_sync
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Dict, Optional, Self, Tuple, TypeAlias

from infrastructure.decorators import run_once_per_instance
from infrastructure.mixins import LogClassMixin
from infrastructure.schemas import (
    DeveloperError,
    FrameTensorMetadata,
    ImagesTensorType,
    PointCloudTensors,
    UnexpectedError,
    VGGTModelOptions,
)

if TYPE_CHECKING:  # allows lazy import of risky modules like vggt
    import torch
    from vggt.models.vggt import VGGT


class VGGTInterface(LogClassMixin):
    instance: ClassVar[Optional[Self]] = None

    InitQueueTuple: TypeAlias = Tuple[str, str, int, int]

    def __new__(cls) -> Self:
        """
        Ensures only while instance of the interface exists at a time.
        Only after cleanup will new instances be created.
        """
        if cls.instance is None:
            cls.instance = super().__new__(cls)

        return cls.instance

    @run_once_per_instance
    def __init__(self):
        import torch

        self.model: Optional[VGGT] = None
        self.hf_id: Optional[str] = None
        self.model_cache_dir: Optional[Path] = None
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype: torch.dtype = torch.float32

    def initialize_model(
        self,
        hf_id: str,
        model_cache_dir: Path,
    ):
        try:
            from vggt.models.vggt import VGGT

            if all([self.model, self.hf_id, self.model_cache_dir]):
                return  # initialization did not occur

            # initialize model from the downloaded local model cache
            self.model = (
                VGGT.from_pretrained(hf_id, cache_dir=model_cache_dir)
                .to(device=self.device)
                .to(dtype=self.dtype)
            )

            # store the values used for initialization upon success
            self.hf_id = hf_id
            self.model_cache_dir = model_cache_dir

        except Exception as e:
            self.cleanup()
            raise e

    @staticmethod
    def download_model(
        hf_id: str,
        model_cache_dir: Path,
        queue: mp.Queue[InitQueueTuple],
        cancel_event: mp_sync.Event,
    ):
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

            hf_hub_download(
                repo_id=hf_id,
                filename=SAFETENSORS_SINGLE_FILE,
                cache_dir=str(model_cache_dir),
                tqdm_class=make_tqdm_class(queue),
            )
            if cancel_event.is_set():
                return
            queue.put(("downloaded", "Download finished.", -1, -1))
        except Exception as e:
            queue.put(("error", str(e), -1, -1))

    def run_inference(
        self,
        images: ImagesTensorType,
        metadata: FrameTensorMetadata,
        options: VGGTModelOptions,
    ) -> PointCloudTensors:
        if self.model is None:
            raise DeveloperError("Model not initialized.")

        import torch
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        images = images.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            predictions: Dict[str, torch.Tensor] = self.model(images)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:], build_intrinsics=True
        )
        if not intrinsic:
            raise UnexpectedError("`build_intrinsics` argument ineffective.")

        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        depth = predictions["depth"]
        depth_conf = predictions["depth_conf"]
        pointmap = predictions["world_points"]
        pointmap_conf = predictions["world_points_conf"]

        if options.mode == "pointmap":
            world_points = pointmap
            conf_map = pointmap_conf
        else:
            world_points = unproject_depth_map_to_point_map(
                depth.cpu().numpy(),
                predictions["extrinsic"].cpu().numpy(),
                predictions["intrinsic"].cpu().numpy(),
            )
            conf_map = depth_conf

        B, H, W, _ = world_points.shape

        xyz = world_points.reshape(-1, 3)
        conf = conf_map.reshape(-1)

        rgb = (images.permute(0, 2, 3, 1).reshape(-1, 3).clamp(0, 1) * 255).to(
            torch.uint8
        )

        cam_idx = torch.repeat_interleave(
            torch.arange(B, device=self.device, dtype=torch.int32),
            H * W,
        )

        if conf.numel() > 0:
            conf_threshold = torch.quantile(conf, options.confidence_percentile / 100.0)
        else:
            conf_threshold = torch.tensor(0.0, device=conf.device)

        mask = (conf >= conf_threshold) & (conf > 1e-5)

        if options.mask_black:
            black_bg_mask = rgb.sum(dim=1) >= 16
            mask &= black_bg_mask

        if options.mask_white:
            white_bg_mask = ~((rgb[:, 0] > 240) & (rgb[:, 1] > 240) & (rgb[:, 2] > 240))
            mask &= white_bg_mask

        xyz = xyz[mask]
        rgb = rgb[mask]
        conf = conf[mask]
        cam_idx = cam_idx[mask]

        xyz = xyz.cpu()
        rgb = rgb.cpu()
        conf = conf.cpu()
        cam_idx = cam_idx.cpu()

        extrinsic = extrinsic.cpu()
        if intrinsic:
            intrinsic = intrinsic.cpu()
        depth = depth.cpu()
        depth_conf = depth_conf.cpu()

        return PointCloudTensors(
            xyz=xyz,
            rgb=rgb,
            conf=conf,
            cam_idx=cam_idx,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            depth=depth,
            depth_conf=depth_conf,
            point_map=pointmap,
            _metadata=metadata,
        )

    def cleanup(self):
        """clean up expensive resources"""
        try:
            if self.model is not None:
                # force CUDA tensors to be released before we release our object
                self.model.to("cpu")
                del self.model
                self.model = None

                self.release_resources()
        except Exception as e:
            raise RuntimeError("Error while cleaning up model.") from e

        self.hf_id = None
        self.model_cache_dir = None

    @staticmethod
    def release_resources():
        from torch import cuda

        if cuda.is_available():
            cuda.synchronize()  # ensure all kernels finish

        cuda.empty_cache()  # only effective when all torch resources have been released
        gc.collect()

    @classmethod
    def cleanup_interface(cls):
        if cls.instance:
            cls.instance.cleanup()
        cls.instance = None  # set instance to none


def make_tqdm_class(queue: mp.Queue[VGGTInterface.InitQueueTuple]):
    from huggingface_hub.utils.tqdm import tqdm

    class ProgressTqdm(tqdm):
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            self._queue = queue
            kwargs["disable"] = False
            super().__init__(*args, **kwargs)

        def display(self, *args, **kwargs):
            if self.total:
                self._queue.put(
                    (
                        "progress",
                        "",
                        int(float(self.n) / 100.0),  # convert from bytes to mb
                        int(float(self.total) / 100.0),
                    )
                )

    return ProgressTqdm
