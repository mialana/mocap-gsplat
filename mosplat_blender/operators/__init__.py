from typing import Dict, List, Tuple, Type

from ..infrastructure.identifiers import OperatorIDEnum
from ..infrastructure.mixins import PreregristrationFn
from ..infrastructure.schemas import AddonMeta
from .base_ot import (
    MosplatOperatorBase,
    MosplatOperatorMetadata,
)
from .extract_frame_range_ot import (
    Mosplat_OT_extract_frame_range,
)
from .initialize_model_ot import Mosplat_OT_initialize_model
from .install_pointcloud_preview_ot import (
    Mosplat_OT_install_point_cloud_preview,
)
from .open_addon_preferences_ot import (
    Mosplat_OT_open_addon_preferences,
)
from .run_inference_ot import Mosplat_OT_run_inference
from .run_preprocess_script_ot import (
    Mosplat_OT_run_preprocess_script,
)
from .train_gaussian_splats_ot import Mosplat_OT_train_gaussian_splats
from .validate_media_statuses_ot import (
    Mosplat_OT_validate_media_statuses,
)

operator_registry: Dict[
    Type[MosplatOperatorBase],
    MosplatOperatorMetadata,
] = {
    Mosplat_OT_initialize_model: MosplatOperatorMetadata(
        bl_idname=OperatorIDEnum.INITIALIZE_MODEL,
        bl_description="Download or load VGGT model weights using Hugging Face.",
    ),
    Mosplat_OT_open_addon_preferences: MosplatOperatorMetadata(
        bl_idname=OperatorIDEnum.OPEN_ADDON_PREFERENCES,
        bl_description=f"Quick navigation to '{AddonMeta().human_readable_name}' saved addon preferences.",
        bl_options={"REGISTER", "MACRO"},
    ),
    Mosplat_OT_validate_media_statuses: MosplatOperatorMetadata(
        bl_idname=OperatorIDEnum.VALIDATE_FILE_STATUSES,
        bl_description="Check frame count, width, and height of all media files found in current media directory.",
    ),
    Mosplat_OT_extract_frame_range: MosplatOperatorMetadata(
        bl_idname=OperatorIDEnum.EXTRACT_FRAME_RANGE,
        bl_description="Extract a frame range from all media files in media directory.",
    ),
    Mosplat_OT_run_preprocess_script: MosplatOperatorMetadata(
        bl_idname=OperatorIDEnum.RUN_PREPROCESS_SCRIPT,
        bl_description="Run current preprocess script on current frame range.",
    ),
    Mosplat_OT_run_inference: MosplatOperatorMetadata(
        bl_idname=OperatorIDEnum.RUN_INFERENCE,
        bl_description="Run VGGT model inference on preprocessed image data. Export point cloud tensors and a PLY file for previewing.",
    ),
    Mosplat_OT_install_point_cloud_preview: MosplatOperatorMetadata(
        bl_idname=OperatorIDEnum.INSTALL_POINT_CLOUD_PREVIEW,
        bl_description="Installs a handler that runs before animation frame changes and imports the corresponding exported point cloud PLY file for the frame.",
    ),
    Mosplat_OT_train_gaussian_splats: MosplatOperatorMetadata(
        bl_idname=OperatorIDEnum.TRAIN_GAUSSIAN_SPLATS,
        bl_description="Train a model to generate a gaussian splat representation of the scene from point cloud tensors.",
    ),
}

operator_factory: List[Tuple[Type[MosplatOperatorBase], PreregristrationFn]] = [
    (cls, cls.preregistration_fn_factory(metadata=mta))
    for cls, mta in operator_registry.items()
]
