"""
provides the interface between the add-on and the VGGT model.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import multiprocessing.synchronize as mp_sync
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Dict, Optional, Self, Tuple, TypeAlias

from ..infrastructure.decorators import run_once_per_instance
from ..infrastructure.macros import crop_tensor, to_0_1, to_0_255
from ..infrastructure.mixins import LogClassMixin
from ..infrastructure.schemas import (
    DeveloperError,
    FrameTensorMetadata,
    ImagesAlphaTensorLike,
    ImagesTensorLike,
    ModelInferenceMode,
    PointCloudTensors,
    VGGTModelOptions,
)

if TYPE_CHECKING:  # allows lazy import of risky modules like vggt
    import torch
    from vggt.models.vggt import VGGT

VGGT_EXPECTED_MAX_SIZE = 518
VGGT_EXPECTED_PIXEL_MULTIPLE = 14


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
        self.model_device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_dtype: torch.dtype = torch.float32

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
                .to(device=self.model_device)
                .to(dtype=self.model_dtype)
                .eval()
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
        images: ImagesTensorLike,
        images_alpha: ImagesAlphaTensorLike,
        metadata: FrameTensorMetadata,
        options: VGGTModelOptions,
    ) -> PointCloudTensors:
        if self.model is None:
            raise DeveloperError("Model not initialized.")

        import torch
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        if TYPE_CHECKING:
            from jaxtyping import Bool, Float32, Int32, UInt8

        images_alpha = to_0_1(images_alpha)
        images = to_0_1(images)  # apply alpha mask

        images = crop_tensor(  # crop down
            images.to(device=self.model_device),
            max_size=VGGT_EXPECTED_MAX_SIZE,
            multiple=VGGT_EXPECTED_PIXEL_MULTIPLE,
        )
        images_alpha = crop_tensor(  # crop down
            images_alpha.to(device=self.model_device),
            max_size=VGGT_EXPECTED_MAX_SIZE,
            multiple=VGGT_EXPECTED_PIXEL_MULTIPLE,
        )
        images = images * images_alpha

        with torch.inference_mode():
            predictions: Dict[str, torch.Tensor] = self.model(images)

        extri_intri = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:], build_intrinsics=True
        )
        assert extri_intri[1] is not None  # specified `build_intrinsics` arg

        extrinsic: Float32[torch.Tensor, "B 3 4"] = extri_intri[0]
        intrinsic: Float32[torch.Tensor, "B 3 3"] = extri_intri[1]

        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        depth: Float32[torch.Tensor, "1 B H W 1"] = predictions["depth"]
        depth_conf: Float32[torch.Tensor, "1 B H W"] = predictions["depth_conf"]
        pointmap: Float32[torch.Tensor, "1 B H W 3"] = predictions["world_points"]
        pointmap_conf: Float32[torch.Tensor, "1 B H W"] = predictions[
            "world_points_conf"
        ]

        world_points: Float32[torch.Tensor, "1 B H W 3"]
        conf_map: Float32[torch.Tensor, "1 B H W"]
        if options.inference_mode == ModelInferenceMode.POINTMAP:
            world_points = pointmap
            conf_map = pointmap_conf
        else:
            world_points = predictions["world_points"]
            conf_map = depth_conf

        S, B, H, W, _ = world_points.shape  # Scene, Batch, Height, Width, Positions
        B = S * B  # flatten first two dimensions

        world_points: Float32[torch.Tensor, "B H W 3"] = world_points.reshape(
            B, H, W, 3
        )
        conf_map: Float32[torch.Tensor, "B H W"] = conf_map.reshape(B, H, W)

        xyz: Float32[torch.Tensor, "N 3"] = world_points.reshape(-1, 3)
        conf: Float32[torch.Tensor, "N"] = conf_map.reshape(-1)

        rgb: UInt8[torch.Tensor, "N 3"] = to_0_255(
            images.permute(0, 2, 3, 1).reshape(-1, 3)
        )

        alpha: Float32[torch.Tensor, "N"] = images_alpha.reshape(-1)

        conf_threshold: Float32[torch.Tensor, "1"] = torch.quantile(
            conf, options.confidence_percentile / 100.0
        )
        mask: Bool[torch.Tensor, "N"] = (conf >= conf_threshold) & (conf > 1e-6)
        mask &= alpha > 0.5  # use alpha as an additional mask

        cam_idx: Int32[torch.Tensor, "N"] = torch.repeat_interleave(
            torch.arange(B, device=conf.device, dtype=torch.int32),
            H * W,
        )

        xyz = xyz[mask].cpu()
        rgb = rgb[mask].cpu()
        conf = conf[mask].cpu()
        cam_idx = cam_idx[mask].cpu()

        extrinsic = extrinsic.reshape(B, 3, 4).cpu()
        intrinsic = intrinsic.reshape(B, 3, 3).cpu()
        depth = depth.reshape(B, H, W, 1).cpu()
        depth_conf = depth_conf.reshape(B, H, W).cpu()
        pointmap = pointmap.reshape(B, H, W, 3).cpu()

        return PointCloudTensors(
            xyz=xyz,
            rgb=rgb,
            conf=conf,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            depth=depth,
            depth_conf=depth_conf,
            pointmap=pointmap,
            cam_idx=cam_idx,
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
