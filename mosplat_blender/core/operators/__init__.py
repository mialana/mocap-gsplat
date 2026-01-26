from typing import List, Type

from .base_ot import MosplatOperatorBase
from . import (
    initialize_model_ot,
    open_addon_preferences_ot,
    run_inference_ot,
    extract_frame_range_ot,
    validate_media_file_statuses_ot,
)

all_operators: List[Type[MosplatOperatorBase]] = [
    initialize_model_ot.Mosplat_OT_initialize_model,
    run_inference_ot.Mosplat_OT_run_inference,
    open_addon_preferences_ot.Mosplat_OT_open_addon_preferences,
    validate_media_file_statuses_ot.Mosplat_OT_validate_media_file_statuses,
    extract_frame_range_ot.Mosplat_OT_extract_frame_range,
]
