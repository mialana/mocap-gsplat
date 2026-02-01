"""
provides the interface between the add-on and the VGGT model.
"""

from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional, final

from ..infrastructure.decorators import no_instantiate
from ..infrastructure.mixins import LogClassMixin
from ..infrastructure.schemas import UnexpectedError

if TYPE_CHECKING:  # allows lazy import of risky modules like vggt
    from vggt.models.vggt import VGGT


@final
@no_instantiate
class MosplatVGGTInterface(LogClassMixin):
    model: ClassVar[Optional[VGGT]] = None
    hf_id: ClassVar[Optional[str]] = None
    cache_dir: ClassVar[Optional[Path]] = None

    @classmethod
    def initialize_model(
        cls,
        hf_id: str,
        model_cache_dir: Path,
        cancel_event: threading.Event,
    ):
        try:
            from vggt.models.vggt import VGGT

            if cls.model:
                return  # initialization did not occur

            # initialize model from the downloaded local model cache
            cls.model = VGGT.from_pretrained(hf_id, cache_dir=model_cache_dir)
            cls.hf_id = hf_id
            cls.cache_dir = model_cache_dir  # store the values used for initialization

            if (
                cancel_event.is_set()
            ):  # if cancel event was set at some point cleanup the resources now
                cls.cleanup()
        except Exception as e:
            cls.cleanup()
            raise UnexpectedError("Error while initializing model.", e) from e

    @classmethod
    def cleanup(cls):
        """clean up expensive resources"""
        try:
            if cls.model is not None:
                cls.model.to(
                    "cpu"
                )  # force CUDA tensors to be released before we release our object
                del cls.model
                cls.model = None

                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # ensure all kernels finish

                torch.cuda.empty_cache()  # only effective when all torch resources have been released
                gc.collect()

                cls.class_logger.info("Cleaned up VGGT model.")

            cls.class_logger.info("Cleaned up worker references")
        except Exception as e:
            raise UnexpectedError("Error while cleaning up model.", e) from e

        cls.hf_id = None
        cls.cache_dir = None
        cls.class_logger.info("Removed initialization variables.")
