"""
provides the interface between the add-on and the VGGT model.
"""

from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING, final, Optional
from pathlib import Path
import threading
import gc

from ..infrastructure.mixins import MosplatLogClassMixin
from ..infrastructure.decorators import no_instantiate
from ..infrastructure.schemas import UnexpectedError

if TYPE_CHECKING:  # allows lazy import of risky modules like vggt
    from vggt.models.vggt import VGGT


@final
@no_instantiate
class MosplatVGGTInterface(MosplatLogClassMixin):
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
            cls.model = VGGT.from_pretrained(
                hf_id, cache_dir=model_cache_dir, local_files_only=True
            )
            cls.hf_id = hf_id
            cls.cache_dir = model_cache_dir  # store the values used for initialization

            if (
                cancel_event.is_set()
            ):  # if cancel event was set at some point cleanup the resources now
                cls.cleanup()
        except Exception as e:
            cls.cleanup()
            raise UnexpectedError(str(e)) from e

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

                cls.logger().info("Cleaned up VGGT model.")

            cls.logger().info("Cleaned up worker references")
        except Exception as e:
            raise UnexpectedError(str(e)) from e

        cls.hf_id = None
        cls.cache_dir = None
        cls.logger().info("Removed initialization variables.")
