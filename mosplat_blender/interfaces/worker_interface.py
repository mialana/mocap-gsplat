from __future__ import annotations

import threading
from queue import Queue, Empty
from typing import Callable, Generic, TypeVar, Optional, final

from ..infrastructure.decorators import no_instantiate

T = TypeVar("T")


class OperatorWorker(Generic[T]):
    """
    worker abstraction for blender ops

    - thread-owned execution
    - queue-based communication
    - event-based cancellation
    - no blender references
    """

    def __init__(self, run_fn: Callable[[Queue, threading.Event], None]):
        self._queue: Queue = Queue()
        self._cancel_event = threading.Event()

        self._thread = threading.Thread(
            target=run_fn,
            args=(self._queue, self._cancel_event),
            daemon=True,
        )

    def start(self):
        self._thread.start()

    def cancel(self):
        self._cancel_event.set()

    def cleanup(self):
        self.cancel()

        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def poll(self) -> Optional[T]:
        try:
            return self._queue.get_nowait()
        except Empty:
            return None


@final
@no_instantiate
class MosplatWorkerInterface:
    @staticmethod
    def create_worker(
        run_fn: Callable[[Queue, threading.Event], None],
    ) -> OperatorWorker:
        return OperatorWorker(run_fn)
