from __future__ import annotations

import multiprocessing as mp
import multiprocessing.synchronize as mp_sync
import os
from datetime import datetime
from queue import Empty
from typing import Callable, Generic, Optional, TypeAlias, TypeVar

QT = TypeVar("QT")  # types elements of worker queue

from infrastructure.constants import _TIMEOUT_IMMEDIATE_
from infrastructure.decorators import record_work_time
from infrastructure.mixins import LogClassMixin
from infrastructure.schemas import EnvVariableEnum

StartWorkCallback: TypeAlias = Optional[Callable[[float], None]]
EndWorkCallback: TypeAlias = Optional[Callable[[float, float], None]]


class MosplatWorkerInterface(Generic[QT], LogClassMixin):
    """
    this is a clean abstraction for blender operators that require similar "worker" behavior in a separate thread

    features execution through threading (and start / end callbacks if necessary),
    a queue + polling mechanism for maintaining communication with main thread,
    and `threading.Event`-based cancellation.
    """

    def __init__(
        self,
        owner_name: str,
        worker_fn: Callable[[mp.Queue[QT], mp_sync.Event], None],
        *,
        use_start_work_callback: bool = True,
        start_work_callback: StartWorkCallback = None,
        use_end_work_callback: bool = True,
        end_work_callback: EndWorkCallback = None,
    ):
        self._ctx = mp.get_context("spawn")

        # create queue and cancel event
        self._queue: mp.Queue[QT] = self._ctx.Queue()
        self._cancel_event: mp_sync.Event = self._ctx.Event()

        # set up start work callback unless opt-ed out
        self._start_work_callback: StartWorkCallback = None
        self._end_work_callback: EndWorkCallback = None

        if use_start_work_callback:
            self._start_work_callback = (
                start_work_callback or self._default_start_work_callback
            )
        if use_end_work_callback:
            self._end_work_callback = (
                end_work_callback or self._default_end_work_callback
            )

        self._process = self._ctx.Process(
            target=worker_fn,
            args=(self._queue, self._cancel_event),
            daemon=True,
        )
        self._id: Optional[int] = None
        self._owner_name: str = owner_name

    @property
    def identifier(self) -> str:
        return str(self._id) if self._id is not None else self._owner_name

    def start(self):
        os.environ.setdefault(EnvVariableEnum.SUBPROCESS_FLAG, "1")
        self._process.start()
        os.environ.pop(EnvVariableEnum.SUBPROCESS_FLAG)
        self._id = self._process.pid

    def cancel(self):
        self._cancel_event.set()
        self.logger.debug(f"Thread '{self.identifier}' was cancelled.")

    def was_cancelled(self):
        return self._cancel_event.is_set()

    def cleanup(self):
        self.cancel()

        if self._process.is_alive():
            self._process.join(timeout=_TIMEOUT_IMMEDIATE_)

        if self._process.is_alive():
            self.logger.warning(
                f"Process '{self.identifier}' did not exit in time and was terminated."
            )
            self._process.terminate()
            self._process.join()

        # drain queue
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break
        self.logger.debug(f"'{self.identifier}' clean.")

    def dequeue(self) -> Optional[QT]:
        try:
            return self._queue.get_nowait()
        except Empty:
            return None

    def _default_start_work_callback(self, start_time: float):
        start: str = datetime.fromtimestamp(start_time).strftime("%H:%M:%S")

        self.logger.info(
            f"Worker process for '{self._owner_name}' started with ID '{self._id}' at time '{start}'."
        )

    def _default_end_work_callback(self, end_time: float, time_elapsed: float):
        end: str = datetime.fromtimestamp(end_time).strftime("%H:%M:%S")

        self.logger.info(
            f"Worker process for '{self._owner_name}' with ID '{self._id}' ended at time '{end}'."
        )

        self.logger.info(f"Total time elapsed: '{time_elapsed:.2f}' seconds")
