from typing import List, Type

from .base_ot import MosplatOperatorBase
from . import install_model_ot, load_images_ot

all_operators: List[Type[MosplatOperatorBase]] = [
    install_model_ot.Mosplat_OT_initialize_model,
    load_images_ot.Mosplat_OT_load_images,
]
