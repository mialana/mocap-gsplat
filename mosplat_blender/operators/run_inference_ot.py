from functools import partial
from pathlib import Path
from typing import List, Literal, NamedTuple, Tuple, TypeAlias, cast

from ..infrastructure.schemas import (
    AppliedPreprocessScript,
    FrameTensorMetadata,
    SavedTensorFileName,
    SavedTensorKey,
    UserAssertionError,
    UserFacingError,
    VGGTModelOptions,
)
from ..interfaces.vggt_interface import VGGTInterface
from .base_ot import MosplatOperatorBase

PlyFileFormat: TypeAlias = Literal["ascii", "binary"]


class ThreadKwargs(NamedTuple):
    preprocess_script: Path
    media_files: List[Path]
    frame_range: Tuple[int, int]
    exported_file_formatter: str
    ply_file_format: PlyFileFormat
    force: bool
    model_options: VGGTModelOptions


class Mosplat_OT_run_inference(MosplatOperatorBase[Tuple[str, str], ThreadKwargs]):
    @classmethod
    def _contexted_poll(cls, pkg):

        if VGGTInterface().model is None:
            cls._poll_error_msg_list.append("Model must be initialized.")
        if not pkg.props.was_frame_range_extracted:
            cls._poll_error_msg_list.append("Frame range must be extracted.")
        if not pkg.props.was_frame_range_preprocessed:
            cls._poll_error_msg_list.append("Frame range must be preprocessed.")

        return len(cls._poll_error_msg_list) == 0

    def _queue_callback(self, pkg, event, next):
        status, _ = next
        if status == "done":
            pkg.props.ran_inference_on_frame_range = True

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
        ply_file_format: PlyFileFormat = cast(
            PlyFileFormat, str(pkg.prefs.ply_file_format)
        )
        self.launch_thread(
            pkg.context,
            twargs=ThreadKwargs(
                preprocess_script=self._preprocess_script,
                media_files=self._media_files,
                frame_range=self._frame_range,
                exported_file_formatter=self._exported_file_formatter,
                ply_file_format=ply_file_format,
                force=bool(pkg.prefs.force_all_operations),
                model_options=pkg.props.options_accessor.to_dataclass(),
            ),
        )

        return "RUNNING_MODAL"

    @staticmethod
    def _operator_thread(queue, cancel_event, *, twargs):

        import torch
        from safetensors.torch import save_file

        from ..infrastructure.dl_ops import (
            PointCloudTensors,
            TensorTypes as TensorTypes,
            load_and_validate_tensor_file,
            save_ply_ascii,
            save_ply_binary,
        )

        INTERFACE = VGGTInterface()

        (
            script,
            files,
            (start, end),
            exported_file_formatter,
            ply_format,
            force,
            options,
        ) = twargs

        pre_file_formatter = partial(
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
            file_name=SavedTensorFileName.POINT_CLOUD,
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
        pct_tensor_map = PointCloudTensors.map()

        applied_preprocess_script = AppliedPreprocessScript.from_file_path(script)

        pre_metadata: FrameTensorMetadata = FrameTensorMetadata(
            -1, files, applied_preprocess_script, None
        )  # options did not exist in 'run preprocess script' step
        pct_metadata: FrameTensorMetadata = FrameTensorMetadata(
            -1, files, applied_preprocess_script, options
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for idx in range(start, end):
            if cancel_event.is_set():
                return

            pre_in_file = Path(pre_file_formatter(frame_idx=idx))
            pct_out_file = Path(pct_file_formatter(frame_idx=idx))
            ply_out_file = Path(ply_file_formatter(frame_idx=idx))

            pre_metadata.frame_idx = idx
            pct_metadata.frame_idx = idx

            try:
                if not force:
                    try:  # try locating prior data on disk
                        _ = load_and_validate_tensor_file(
                            pct_out_file, device, pct_metadata, map=pct_tensor_map
                        )
                        queue.put(
                            (
                                "update",
                                f"Previous point cloud inference data found on disk for frame '{idx}'",
                            )
                        )
                        continue  # skip this frame
                    except (OSError, UserAssertionError, UserFacingError) as e:
                        pass  # data on disk is not valid

                pre_tensors = load_and_validate_tensor_file(
                    pre_in_file, device, pre_metadata, map=pre_tensor_map
                )
                images: TensorTypes.ImagesTensor_0_255 = pre_tensors[
                    SavedTensorKey.IMAGES
                ]
                images_alpha: TensorTypes.ImagesAlphaTensor_0_255 = pre_tensors[
                    SavedTensorKey.IMAGES_ALPHA
                ]

                pct: PointCloudTensors = INTERFACE.run_inference(
                    images, images_alpha, options
                )
                # save point cloud tensors to disk
                save_file(pct.to_dict(), pct_out_file, metadata=pct_metadata.to_dict())
                # save PLY file to disk
                if ply_format == "ascii":
                    save_ply_ascii(ply_out_file, pct.xyz, pct.rgb_0_255)
                else:
                    save_ply_binary(ply_out_file, pct.xyz, pct.rgb_0_255)

                queue.put(("update", f"Ran inference on frame '{idx}'"))
            except Exception as e:
                msg = UserFacingError.make_msg(
                    f"Error ocurred while running inference on frame '{idx}'.", e
                )
                queue.put(("warning", msg))

        queue.put(("done", "Inference complete for current frame range."))
