"""
`core` is suitable for making `__init__.py` be an import hub,
as it has a very finite amount of imports that need to all be imported
for addon registration.

This is opposed to `infrastructure`, where `__init__.py` is empty, and individual modules files should be imported as needed.
"""

from typing import List, Tuple, Type

from ..infrastructure.mixins import PreregristrationFn
from . import checks, handlers
from .panels import (
    MosplatPanelBase,
    MosplatUIListBase,
    panel_factory,
    ui_list_factory,
)
from .preferences import Mosplat_AP_Global
from .properties import (
    Mosplat_PG_AppliedPreprocessScript,
    Mosplat_PG_Global,
    Mosplat_PG_LogEntry,
    Mosplat_PG_LogEntryHub,
    Mosplat_PG_MediaFileStatus,
    Mosplat_PG_MediaIOMetadata,
    Mosplat_PG_OperatorProgress,
    Mosplat_PG_ProcessedFrameRange,
    Mosplat_PG_SplatTrainingConfig,
    Mosplat_PG_VGGTModelOptions,
    MosplatPropertyGroupBase,
)

preferences_factory: Tuple[Type[Mosplat_AP_Global], PreregristrationFn] = (
    Mosplat_AP_Global,
    Mosplat_AP_Global.preregistration_fn_factory(),
)

# property groups need to be registered in a bottom-to-top "owning" class order
properties_registry: List[Type[MosplatPropertyGroupBase]] = [
    Mosplat_PG_AppliedPreprocessScript,
    Mosplat_PG_ProcessedFrameRange,
    Mosplat_PG_MediaFileStatus,
    Mosplat_PG_MediaIOMetadata,
    Mosplat_PG_OperatorProgress,
    Mosplat_PG_LogEntry,
    Mosplat_PG_LogEntryHub,
    Mosplat_PG_VGGTModelOptions,
    Mosplat_PG_SplatTrainingConfig,
    Mosplat_PG_Global,
]

properties_factory: List[Tuple[Type[MosplatPropertyGroupBase], PreregristrationFn]] = [
    (cls, cls.preregistration_fn_factory()) for cls in properties_registry
]
