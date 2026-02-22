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
from .base_ot import MosplatOperatorBase

PlyFileFormat: TypeAlias = Literal["ascii", "binary"]


class ThreadKwargs(NamedTuple):
    preprocess_script: Path
    media_files: List[Path]
    frame_range: Tuple[int, int]
    exported_file_formatter: str
    ply_file_format: PlyFileFormat
    model_options: VGGTModelOptions


class Mosplat_OT_run_inference(MosplatOperatorBase[Tuple[str, str], ThreadKwargs]):
    @classmethod
    def _contexted_poll(cls, pkg):
        from ..interfaces.vggt_interface import VGGTInterface

        if VGGTInterface().model is None:
            cls._poll_error_msg_list.append("Model must be initialized.")
        if not pkg.props.was_frame_range_extracted:
            cls._poll_error_msg_list.append("Frame range must be extracted.")

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
            TensorTypes as TT,
            load_and_verify_tensor_file,
            save_ply_ascii,
            save_ply_binary,
        )
        from ..interfaces.vggt_interface import VGGTInterface

        INTERFACE = VGGTInterface()

        script, files, (start, end), exported_file_formatter, format, options = twargs
        in_file_formatter = partial(
            exported_file_formatter.format,
            file_name=SavedTensorFileName.PREPROCESSED,
            file_ext="safetensors",
        )
        out_file_formatter = partial(
            exported_file_formatter.format,
            file_name=SavedTensorFileName.MODEL_INFERENCE,
            file_ext="safetensors",
        )
        ply_file_formatter = partial(
            exported_file_formatter.format,
            file_name=SavedTensorFileName.POINTCLOUD,
            file_ext="ply",
        )

        image_tensor_map = {
            SavedTensorKey.IMAGES.value: TT.annotation_of(TT.ImagesTensor_0_255),
            SavedTensorKey.IMAGES_ALPHA.value: TT.annotation_of(
                TT.ImagesAlphaTensor_0_255
            ),
        }
        pointcloud_tensor_map = PointCloudTensors.map()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        applied_preprocess_script = AppliedPreprocessScript.from_file_path(script)

        for idx in range(start, end):
            if cancel_event.is_set():
                return

            tensor_in_file = Path(in_file_formatter(frame_idx=idx))
            tensor_out_file = Path(out_file_formatter(frame_idx=idx))
            ply_out_file = Path(ply_file_formatter(frame_idx=idx))

            new_metadata: FrameTensorMetadata = FrameTensorMetadata(
                idx, files, applied_preprocess_script, options
            )

            pc_tensors: PointCloudTensors
            try:

                try:
                    _ = load_and_verify_tensor_file(
                        tensor_out_file,
                        device,
                        new_metadata,
                        map=pointcloud_tensor_map,
                    )
                    queue.put(
                        (
                            "update",
                            f"Previous point cloud inference data found on disk for frame '{idx}'",
                        )
                    )
                except (OSError, UserAssertionError, UserFacingError) as e:
                    validation_metadata: FrameTensorMetadata = FrameTensorMetadata(
                        idx,
                        files,
                        preprocess_script=applied_preprocess_script,
                        model_options=None,
                    )  # options did not exist in 'run preprocess script' step
                    tensors = load_and_verify_tensor_file(
                        tensor_in_file,
                        device,
                        validation_metadata,
                        map=image_tensor_map,
                    )
                    images: TT.ImagesTensor_0_255 = tensors[SavedTensorKey.IMAGES]
                    images_alpha: TT.ImagesAlphaTensor_0_255 = tensors[
                        SavedTensorKey.IMAGES_ALPHA
                    ]

                    pc_tensors = INTERFACE.run_inference(images, images_alpha, options)
                    # save point cloud tensors to disk
                    save_file(
                        pc_tensors.to_dict(),
                        tensor_out_file,
                        metadata=new_metadata.to_dict(),
                    )
                    # save PLY file to disk
                    if format == "ascii":
                        save_ply_ascii(
                            ply_out_file, pc_tensors.xyz, pc_tensors.rgb_0_255
                        )
                    else:
                        save_ply_binary(
                            ply_out_file, pc_tensors.xyz, pc_tensors.rgb_0_255
                        )

                    queue.put(("update", f"Ran inference on frame '{idx}'"))
            except Exception as e:
                msg = UserFacingError.make_msg(
                    f"Error ocurred while running inference on frame '{idx}'.", e
                )
                queue.put(("warning", msg))

        queue.put(("done", "Inference complete for current frame range."))
