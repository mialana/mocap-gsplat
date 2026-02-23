from functools import partial
from pathlib import Path
from typing import List, NamedTuple, Tuple

from ..infrastructure.schemas import (
    AppliedPreprocessScript,
    FrameTensorMetadata,
    SavedTensorFileName,
    SavedTensorKey,
    SplatTrainingConfig,
    UserAssertionError,
    UserFacingError,
    VGGTModelOptions,
)
from ..interfaces.vggt_interface import VGGTInterface
from .base_ot import MosplatOperatorBase


class ProcessKwargs(NamedTuple):
    preprocess_script: Path
    media_files: List[Path]
    frame_range: Tuple[int, int]
    exported_file_formatter: str
    median_height: int
    median_width: int
    model_options: VGGTModelOptions
    training_config: SplatTrainingConfig


class Mosplat_OT_train_gaussian_splats(
    MosplatOperatorBase[Tuple[str, str], ProcessKwargs]
):
    @classmethod
    def _contexted_poll(cls, pkg):

        if VGGTInterface().model is None:
            cls._poll_error_msg_list.append("Model must be initialized.")
        if not pkg.props.was_frame_range_extracted:
            cls._poll_error_msg_list.append("Frame range must be extracted.")
        if not pkg.props.was_frame_range_preprocessed:
            cls._poll_error_msg_list.append("Frame range must be preprocessed.")
        if not pkg.props.ran_inference_on_frame_range:
            cls._poll_error_msg_list.append("Must run inference on frame range.")

        return len(cls._poll_error_msg_list) == 0

    def _queue_callback(self, pkg, event, next):

        return super()._queue_callback(pkg, event, next)

    def _contexted_invoke(self, pkg, event):
        prefs = pkg.prefs
        props = pkg.props

        self._media_files: List[Path] = props.media_files(prefs)
        self._frame_range: Tuple[int, int] = props.frame_range_
        self._preprocess_script: Path = prefs.preprocess_media_script_file_

        self._exported_file_formatter: str = props.exported_file_formatter(prefs)
        return self.execute_with_package(pkg)

    def _contexted_execute(self, pkg):
        media_io = pkg.props.media_io_accessor

        self.launch_subprocess(
            pkg.context,
            pwargs=ProcessKwargs(
                preprocess_script=self._preprocess_script,
                media_files=self._media_files,
                frame_range=self._frame_range,
                exported_file_formatter=self._exported_file_formatter,
                median_height=int(media_io.median_height),
                median_width=int(media_io.median_width),
                model_options=pkg.props.options_accessor.to_dataclass(),
                training_config=pkg.props.config_accessor.to_dataclass(),
            ),
        )

        return "RUNNING_MODAL"

    @staticmethod
    def _operator_subprocess(queue, cancel_event, *, pwargs):
        import torch

        from ..infrastructure.dl_ops import (
            PointCloudTensors,
            TensorTypes as TensorTypes,
            load_and_verify_tensor_file,
        )
        from ..interfaces.splat_interface import (
            GsplatRasterizer,
            SplatModel,
            scalars_from_xyz,
            train_3dgs,
        )

        script, files, (start, end), exported_file_formatter, H, W, options, config = (
            pwargs
        )
        images_file_formatter = partial(
            exported_file_formatter.format,
            file_name=SavedTensorFileName.PREPROCESSED,
            file_ext="safetensors",
        )
        pct_file_formatter = partial(
            exported_file_formatter.format,
            file_name=SavedTensorFileName.POINT_CLOUD_TENSORS,
            file_ext="safetensors",
        )
        ply_file_formatter = partial(
            exported_file_formatter.format,
            file_name=SavedTensorFileName.SPLAT,
            file_ext="ply",
        )

        pre_tensor_map = {
            SavedTensorKey.IMAGES.value: TensorTypes.annotation_of(
                TensorTypes.ImagesTensor_0_255
            ),
            SavedTensorKey.IMAGES_ALPHA.value: TensorTypes.annotation_of(
                TensorTypes.ImagesAlphaTensor_0_255
            ),
        }
        pct_map = PointCloudTensors.map()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        applied_preprocess_script = AppliedPreprocessScript.from_file_path(script)

        pre_metadata: FrameTensorMetadata = FrameTensorMetadata(
            -1, files, applied_preprocess_script, None
        )  # options did not exist in 'run preprocess script' step
        pct_metadata: FrameTensorMetadata = FrameTensorMetadata(
            -1, files, applied_preprocess_script, options
        )

        rasterizer = GsplatRasterizer(device, (H, W), sh_degree=config.sh_degree)

        for idx in range(start, end):
            if cancel_event.is_set():
                return

            pre_in_file = Path(images_file_formatter(frame_idx=idx))
            pct_in_file = Path(pct_file_formatter(frame_idx=idx))
            ply_out_file = Path(ply_file_formatter(frame_idx=idx))

            pre_metadata.frame_idx = idx
            pct_metadata.frame_idx = idx

            try:
                try:
                    pct_dict = load_and_verify_tensor_file(
                        pct_in_file, device, pct_metadata, map=pct_map
                    )
                    pct: PointCloudTensors = PointCloudTensors.from_dict(pct_dict)

                    pre_tensors = load_and_verify_tensor_file(
                        pre_in_file, device, pre_metadata, map=pre_tensor_map
                    )
                    images: TensorTypes.ImagesTensor_0_255 = pre_tensors[
                        SavedTensorKey.IMAGES
                    ]
                    images_alpha: TensorTypes.ImagesAlphaTensor_0_255 = pre_tensors[
                        SavedTensorKey.IMAGES_ALPHA
                    ]
                except (OSError, UserAssertionError, UserFacingError) as e:
                    msg = UserFacingError.make_msg(
                        f"Could not load saved data from disk for frame '{idx}'. Re-run preprocess step and inference step to clean up data state.",
                        e,
                    )
                    queue.put(("error", msg))
                    return  # exit early here

                voxel_size, base_scale = scalars_from_xyz(pct.xyz)
                model = SplatModel.init_from_pointcloud_tensors(
                    pct,
                    device,
                    voxel_size=voxel_size,
                    base_scale=base_scale,
                    sh_degree=config.sh_degree,
                )

                train_3dgs(
                    model, rasterizer, pct, images, images_alpha, ply_out_file, config
                )

            except Exception as e:
                msg = UserFacingError.make_msg(
                    f"Error ocurred while training gaussian splats on frame '{idx}'.", e
                )
                queue.put(("warning", msg))

        queue.put(("done", "Training complete for current frame range."))


def process_entrypoint(*args, **kwargs):
    Mosplat_OT_train_gaussian_splats._operator_subprocess(*args, **kwargs)
