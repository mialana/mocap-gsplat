from typing import List, Type

from .base_ot import MosplatOperatorBase
from . import install_model_ot, open_addon_preferences_ot, select_media_directory_ot

all_operators: List[Type[MosplatOperatorBase]] = [
    install_model_ot.Mosplat_OT_initialize_model,
    select_media_directory_ot.Mosplat_OT_select_media_directory,
    open_addon_preferences_ot.Mosplat_OT_open_addon_preferences,
]
