from typing import List, Type

from .base_ot import Mosplat_OT_Base, MosplatOperatorMixin
from . import logging_ot

all_operators: List[Type[Mosplat_OT_Base]] = [logging_ot.OT_InitLogging]
