"""
provides the interface between the add-on and the VGGT model.
"""

from typing import ClassVar, TYPE_CHECKING, TypeAlias
from pathlib import Path
import gc

from .logging_interface import MosplatLoggingInterface

if TYPE_CHECKING:  # allows lazy import of risky modules like vggt
    from vggt.models.vggt import VGGT

    VGGTType: TypeAlias = VGGT
else:
    VGGTType: TypeAlias = object

logger = MosplatLoggingInterface.configure_logger_instance(__name__)


class MosplatVGGTInterface:
    _model: ClassVar[VGGTType | None] = None
    _initialized: ClassVar[bool] = False

    @classmethod
    def initialize_model(cls, hf_id: str, outdir: Path) -> bool:
        from vggt.models.vggt import VGGT

        if cls._model:
            return False  # initialization did not occur

        cls._model = VGGT.from_pretrained(hf_id, cache_dir=outdir)

        cls._initialized = True
        logger.info("VGGT model successfully initialized")

        return cls._initialized

    @classmethod
    def cleanup(cls):
        """clean up expensive resources"""

        if cls._model:
            cls._model.to(
                "cpu"
            )  # force CUDA tensors to be released before we release our object
            del cls._model
            cls._model = None

        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # ensure all kernels finish

        torch.cuda.empty_cache()  # only effective when all torch resources have been released
        gc.collect()

        logger.info("Cleaned up VGGT interface")
