from typing import List, Type, Tuple, Dict, Callable, Generator

from .base_ot import MosplatOperatorBase, MosplatOperatorMetadata

from .initialize_model_ot import Mosplat_OT_initialize_model
from .open_addon_preferences_ot import Mosplat_OT_open_addon_preferences
from .extract_frame_range_ot import Mosplat_OT_extract_frame_range
from .validate_media_file_statuses_ot import Mosplat_OT_validate_media_file_statuses
from .run_preprocess_script_ot import Mosplat_OT_run_preprocess_script

from ...infrastructure.schemas import OperatorIDEnum
from ...infrastructure.constants import ADDON_HUMAN_READABLE

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
        bl_description=f"Quick navigation to '{ADDON_HUMAN_READABLE}' saved addon preferences.",
        bl_options={"REGISTER", "MACRO"},
    ),
    Mosplat_OT_validate_media_file_statuses: MosplatOperatorMetadata(
        bl_idname=OperatorIDEnum.VALIDATE_MEDIA_FILE_STATUSES,
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
}

operator_factory: List[Tuple[Type[MosplatOperatorBase], Callable[[], None]]] = [
    (cls, cls.preregistration_fn_factory(metadata))
    for cls, metadata in operator_registry.items()
]
