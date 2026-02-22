"""
provides the interface between the add-on and the VGGT model.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import multiprocessing.synchronize as mp_sync
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional, Self, Tuple, TypeAlias

from ..infrastructure.constants import VGGT_IMAGE_DIMS_FACTOR, VGGT_MAX_IMAGE_SIZE
from ..infrastructure.decorators import run_once_per_instance
from ..infrastructure.macros import to_0_1, to_0_255, to_channel_as_item
from ..infrastructure.mixins import LogClassMixin
from ..infrastructure.schemas import (
    DeveloperError,
    FrameTensorMetadata,
    ModelInferenceMode,
    PointCloudTensors,
    TensorTypes as TT,
    VGGTModelOptions,
)

if TYPE_CHECKING:  # allows lazy import of risky modules like vggt
    from typing import TypedDict

    from jaxtyping import Bool, Float32, Int32, UInt8
    from torch import Tensor
    from vggt.models.vggt import VGGT as VGGTType

    class VGGTPredictions(TypedDict):
        pose_enc: Float32[Tensor, "B S 9"]
        depth: Float32[Tensor, "B S H W 1"]
        depth_conf: Float32[Tensor, "B S H W"]
        world_points: Float32[Tensor, "B S H W 3"]
        world_points_conf: Float32[Tensor, "B S H W"]
        images: TT.ImagesAlphaTensor_0_1


def ensure_tensor_shape_for_vggt(tensor: Tensor):
    """ensure tensors are the correct shape for VGGT model"""
    assert len(tensor.shape) == 4
    _, _, H, W = tensor.shape
    assert H <= VGGT_MAX_IMAGE_SIZE and W <= VGGT_MAX_IMAGE_SIZE
    assert H % VGGT_IMAGE_DIMS_FACTOR == 0 and W % VGGT_IMAGE_DIMS_FACTOR == 0


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

        self.model: Optional[VGGTType] = None
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
        images_0_255: TT.ImagesAlphaTensor_0_255,
        images_alpha_0_255: TT.ImagesAlphaTensor_0_255,
        options: VGGTModelOptions,
    ) -> PointCloudTensors:
        if self.model is None:
            raise DeveloperError("Model not initialized.")

        import torch
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        ensure_tensor_shape_for_vggt(images_0_255)
        ensure_tensor_shape_for_vggt(images_alpha_0_255)

        images = to_0_1(images_0_255)
        images_alpha = to_0_1(images_alpha_0_255)

        images.mul_(images_alpha)  # apply alpha tensor

        with torch.inference_mode():
            predictions: VGGTPredictions = self.model(images)

        extri_intri = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:], build_intrinsics=True
        )
        assert extri_intri[1] is not None  # specified `build_intrinsics` arg

        extri: Float32[torch.Tensor, "S 3 4"] = extri_intri[0]
        intri: Float32[torch.Tensor, "S 3 3"] = extri_intri[1]

        B, S, H, W, _ = predictions[
            "world_points"
        ].shape  # batch, scene, height, width, positions
        S = B * S  # we are not using batch dimension, so flatten first two dimensions

        extrinsic: TT.ExtrinsicTensor = extri.reshape(S, 3, 4)
        intrinsic: TT.IntrinsicTensor = intri.reshape(S, 3, 3)

        depth: TT.DepthTensor = predictions["depth"].reshape(S, H, W, 1)
        depth_conf: TT.DepthConfTensor = predictions["depth"].reshape(S, H, W)
        pointmap: TT.PointmapTensor = predictions["world_points"].reshape(S, H, W, 3)
        pointmap_conf: TT.PointmapConfTensor = predictions["world_points_conf"].reshape(
            S, H, W
        )

        selected_conf: Float32[torch.Tensor, "S H W"]
        if options.inference_mode == ModelInferenceMode.POINTMAP:
            selected_conf = pointmap_conf
        else:
            selected_conf = depth_conf

        xyz: TT.XYZTensor = pointmap.reshape(-1, 3)
        rgb: TT.RGBTensor = to_0_255(to_channel_as_item(images).reshape(-1, 3))
        conf: TT.ConfTensor = selected_conf.reshape(-1)
        conf_threshold: Float32[torch.Tensor, ""] = torch.quantile(
            conf, options.confidence_percentile / 100.0
        )
        point_cams: Int32[torch.Tensor, "N"] = torch.repeat_interleave(
            torch.arange(S, device=conf.device, dtype=torch.int32),
            H * W,
        )

        alpha: Float32[torch.Tensor, "N"] = images_alpha.reshape(-1)

        mask: Bool[torch.Tensor, "N"] = (conf >= conf_threshold) & (conf > 1e-6)
        mask &= alpha > 0.5  # use alpha as an additional mask

        xyz = xyz[mask]
        rgb = rgb[mask]
        conf = conf[mask]
        point_cams = point_cams[mask]

        return PointCloudTensors(
            xyz=xyz,
            rgb=rgb,
            conf=conf,
            point_cams=point_cams,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            depth=depth,
            depth_conf=depth_conf,
            pointmap=pointmap,
            pointmap_conf=pointmap_conf,
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
