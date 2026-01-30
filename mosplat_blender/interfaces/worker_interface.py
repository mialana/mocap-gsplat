from __future__ import annotations

import threading
from queue import Queue, Empty
from typing import Generic, TypeVar, Optional, Callable, Optional

QT = TypeVar("QT")  # types elements of worker queue

from ..infrastructure.mixins import MosplatLogClassMixin


class MosplatWorkerInterface(Generic[QT], MosplatLogClassMixin):
    """
    an abstraction for blender operators that utilize similar "worker" behavior

    - thread-owned execution
    - queue-based communication
    - event-based cancellation
    - no blender references
    """

    def __init__(
        self, owner_name: str, worker_fn: Callable[[Queue[QT], threading.Event], None]
    ):
        self._queue: Queue[QT] = Queue[QT]()
        self._cancel_event: threading.Event = threading.Event()

        self._thread = threading.Thread(
            target=worker_fn,
            args=(self._queue, self._cancel_event),
            daemon=True,
        )
        self._id: Optional[int] = None

        self._owner_name: str = owner_name

    @property
    def identifier(self) -> str:
        return str(self._id) if self._id else self._owner_name

    def start(self):
        self._thread.start()
        self._id = self._thread.native_id

        self.logger.info(
            f"Worker thread for '{self._owner_name}' started with ID: '{self._id}'"
        )

        def watch():
            self._thread.join()

            self.logger.info(f"Thread '{self.identifier}' completed.")

        # start another daemon thread to watch the worker thread
        threading.Thread(target=watch, daemon=True).start()

    def cancel(self):
        self._cancel_event.set()
        self.logger.debug(f"Thread '{self.identifier}' was cancelled.")

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
        self.logger.debug(f"'{self.identifier}' clean.")

    def dequeue(self) -> Optional[QT]:
        try:
            return self._queue.get_nowait()
        except Empty:
            return None
