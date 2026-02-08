"""
provides the interface between the add-on and the VGGT model.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import multiprocessing.synchronize as mp_sync
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional, Self, Tuple, TypeAlias

from infrastructure.decorators import run_once_per_instance
from infrastructure.mixins import LogClassMixin

if TYPE_CHECKING:  # allows lazy import of risky modules like vggt
    import numpy as np
    from vggt.models.vggt import VGGT


class VGGTInterface(LogClassMixin):
    instance: ClassVar[Optional[Self]] = None

    InitQueueTuple: TypeAlias = Tuple[str, str, int, int]

    def __new__(cls) -> Self:
        """
        Ensures only while instance of the interface exists at a time.
        Only after cleanup will new instances be created.
        """
        if cls.instance is None:
            cls.instance = super().__new__(cls)

        return cls.instance

    @run_once_per_instance
    def __init__(self):
        self.model: Optional[VGGT] = None
        self.hf_id: Optional[str] = None
        self.model_cache_dir: Optional[Path] = None

    def initialize_model(
        self,
        hf_id: str,
        model_cache_dir: Path,
    ):
        try:
            from vggt.models.vggt import VGGT

            if all([self.model, self.hf_id, self.model_cache_dir]):
                return  # initialization did not occur

            # initialize model from the downloaded local model cache
            self.model = VGGT.from_pretrained(hf_id, cache_dir=model_cache_dir)

            # store the values used for initialization upon success
            self.hf_id = hf_id
            self.model_cache_dir = model_cache_dir

        except Exception as e:
            self.cleanup()
            raise e

    @staticmethod
    def download_model(
        hf_id: str,
        model_cache_dir: Path,
        queue: mp.Queue[InitQueueTuple],
        cancel_event: mp_sync.Event,
    ):
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

            hf_hub_download(
                repo_id=hf_id,
                filename=SAFETENSORS_SINGLE_FILE,
                cache_dir=str(model_cache_dir),
                tqdm_class=make_tqdm_class(queue),
            )
            if cancel_event.is_set():
                return
            queue.put(("done", "Download finished.", -1, -1))
        except Exception as e:
            queue.put(("error", str(e), -1, -1))

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
        except Exception as e:
            raise RuntimeError("Error while cleaning up model.") from e

        self.hf_id = None
        self.cache_dir = None

    @classmethod
    def cleanup_interface(cls):
        if cls.instance:
            cls.instance.cleanup()
        cls.instance = None  # set instance to none


def make_tqdm_class(queue: mp.Queue[VGGTInterface.InitQueueTuple]):
    from huggingface_hub.utils.tqdm import tqdm

    class ProgressTqdm(tqdm):
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            self._queue = queue
            kwargs["disable"] = False
            super().__init__(*args, **kwargs)

        def display(self, *args, **kwargs):
            if self.total:
                self._queue.put(
                    (
                        "progress",
                        "",
                        int(float(self.n) / 100.0),  # convert from bytes to mb
                        int(float(self.total) / 100.0),
                    )
                )

    return ProgressTqdm
