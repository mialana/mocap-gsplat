"""
`core` is suitable for making `__init__.py` be an import hub,
as it has a very finite amount of imports that need to all be imported
for addon registration.

This is opposed to `infrastructure`, where `__init__.py` is empty, and individual modules files should be imported as needed.
"""

from .properties import Mosplat_PG_Global
from .preferences import Mosplat_AP_Global
from .operators import MosplatOperatorBase, all_operators
from .panels import MosplatPanelBase, all_panels
