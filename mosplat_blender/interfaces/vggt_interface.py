"""
provides the interface between Blender and the VGGT model.
"""

import os
import gc
import glob
from typing import Optional


class MosplatVGGTInterface:
    def __init__(self):
        self.model = None
        self.device = None
        self.dtype = None
        self._initialized = False
