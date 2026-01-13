from typing import List, Type

from .base import MosplatOperatorBase
from . import install_model_ot

all_operators: List[Type[MosplatOperatorBase]] = [
    install_model_ot.Mosplat_OT_install_model
]
