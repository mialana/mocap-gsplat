"""
provides the interface between the add-on and the VGGT model.
"""

from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional, Self

from ..infrastructure.mixins import LogClassMixin
from ..infrastructure.schemas import UnexpectedError

if TYPE_CHECKING:  # allows lazy import of risky modules like vggt
    import numpy as np
    from vggt.models.vggt import VGGT


class MosplatVGGTInterface(LogClassMixin):
    instance: ClassVar[Optional[Self]] = None

    def __new__(cls) -> Self:
        """
        Ensures only while instance of the interface exists at a time.
        Only after cleanup will new instances be created.
        """
        if cls.instance is None:
            cls.instance = super().__new__(cls)

        return cls.instance

    def __init__(self):
        self.model: Optional[VGGT] = None
        self.hf_id: Optional[str] = None
        self.model_cache_dir: Optional[Path] = None

    def initialize_model(
        self,
        hf_id: str,
        model_cache_dir: Path,
        cancel_event: threading.Event,
    ):
        try:
            from vggt.models.vggt import VGGT

            if all([self.model, self.hf_id, self.model_cache_dir]):
                self.logger.warning("Initialization already occurred.")
                return  # initialization did not occur
            elif any([self.model, self.hf_id, self.model_cache_dir]):
                self.logger.warning(
                    "Model seems to be partially initialized. "
                    "Cleaning up before re-trying initialization."
                )
                self.cleanup()

            # initialize model from the downloaded local model cache
            self.model = VGGT.from_pretrained(hf_id, cache_dir=model_cache_dir)

            self.logger.info("Initialization finished.")

            # store the values used for initialization upon success
            self.hf_id = hf_id
            self.model_cache_dir = model_cache_dir

            if cancel_event.is_set():
                self.logger.warning("Cancellation occurred during initialization.")
                # if cancel event was set at some point cleanup the resources now
                self.cleanup()

        except Exception as e:
            self.cleanup()
            raise UnexpectedError(
                "Error while initializing model."
                f"\nHugging Face ID: '{hf_id}'"
                f"\nModel Cache Dir: '{model_cache_dir}'",
                e,
            ) from e

    def run_inference(self, np_data: np.ndarray):
        from torch import from_numpy, no_grad

        if not self.model:
            return

        tensors = from_numpy(np_data)

        with no_grad():
            predictions = self.model(tensors)

        pass

    def cleanup(self):
        """clean up expensive resources"""
        try:
            if self.model is not None:
                # force CUDA tensors to be released before we release our object
                self.model.to("cpu")
                del self.model
                self.model = None

                from torch import cuda

                if cuda.is_available():
                    cuda.synchronize()  # ensure all kernels finish

                cuda.empty_cache()  # only effective when all torch resources have been released
                gc.collect()

                self.logger.info("Cleaned up model.")
        except Exception as e:
            raise UnexpectedError("Error while cleaning up model.", e) from e

        self.hf_id = None
        self.cache_dir = None

        self.logger.info("Removed initialization variables.")

    @classmethod
    def cleanup_interface(cls):
        if cls.instance:
            cls.instance.cleanup()
        cls.instance = None  # set instance to none
