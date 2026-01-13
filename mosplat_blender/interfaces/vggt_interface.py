"""
provides the interface between Blender and the VGGT model.
"""

from typing import ClassVar, TYPE_CHECKING, Any, TypeAlias
from pathlib import Path


class MosplatVGGTInterface:

    _model: ClassVar = None

    @classmethod
    def initialize_model(cls, hf_id: str, outdir: Path):
        from vggt.models.vggt import VGGT

        if cls._model:
            return cls._model

        cls._model = VGGT.from_pretrained(hf_id, cache_dir=outdir)

        return cls._model
