from __future__ import annotations

import threading
from queue import Queue, Empty
from typing import Generic, TypeVar, Optional, Callable

QT = TypeVar("QT")  # types elements of worker queue for `worker_fn_auto`


class MosplatWorkerInterface(Generic[QT]):
    """
    an abstraction for blender operators that utilize similar "worker" behavior

    - thread-owned execution
    - queue-based communication
    - event-based cancellation
    - no blender references
    """

    def __init__(self, worker_fn: Callable[[Queue[QT], threading.Event], None]):
        self._queue: Queue[QT] = Queue[QT]()
        self._cancel_event = threading.Event()

        self._thread = threading.Thread(
            target=worker_fn,
            args=(self._queue, self._cancel_event),
            daemon=True,
        )

    def start(self):
        self._thread.start()

    def cancel(self):
        self._cancel_event.set()

    def was_cancelled(self):
        return self._cancel_event.is_set()

    def cleanup(self):
        self.cancel()

        # drain queue
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def dequeue(self) -> Optional[QT]:
        try:
            return self._queue.get_nowait()
        except Empty:
            return None
