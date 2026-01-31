from typing import List, Type

from .base_ot import MosplatOperatorBase

from .initialize_model_ot import Mosplat_OT_initialize_model
from .open_addon_preferences_ot import Mosplat_OT_open_addon_preferences
from .run_inference_ot import Mosplat_OT_run_inference
from .extract_frame_range_ot import Mosplat_OT_extract_frame_range
from .validate_media_file_statuses_ot import Mosplat_OT_validate_media_file_statuses

all_operators: List[Type[MosplatOperatorBase]] = [
    Mosplat_OT_initialize_model,
    Mosplat_OT_open_addon_preferences,
    Mosplat_OT_run_inference,
    Mosplat_OT_extract_frame_range,
    Mosplat_OT_validate_media_file_statuses,
]
