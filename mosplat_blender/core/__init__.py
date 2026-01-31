"""
`core` is suitable for making `__init__.py` be an import hub,
as it has a very finite amount of imports that need to all be imported
for addon registration.

This is opposed to `infrastructure`, where `__init__.py` is empty, and individual modules files should be imported as needed.
"""

from typing import List, Type

from .operators import MosplatOperatorBase, all_operators
from .panels import MosplatPanelBase, MosplatUIListBase, all_panels, all_ui_lists
from .preferences import Mosplat_AP_Global
from .properties import (
    MosplatPropertyGroupBase,
    Mosplat_PG_AppliedPreprocessScript,
    Mosplat_PG_ProcessedFrameRange,
    Mosplat_PG_MediaFileStatus,
    Mosplat_PG_MediaIODataset,
    Mosplat_PG_LogEntry,
    Mosplat_PG_OperatorProgress,
    Mosplat_PG_Global,
)

# property groups need to be registered in a bottom-to-top "owning" class order
all_properties: List[Type[MosplatPropertyGroupBase]] = [
    Mosplat_PG_AppliedPreprocessScript,
    Mosplat_PG_ProcessedFrameRange,
    Mosplat_PG_MediaFileStatus,
    Mosplat_PG_MediaIODataset,
    Mosplat_PG_LogEntry,
    Mosplat_PG_OperatorProgress,
    Mosplat_PG_Global,
]
