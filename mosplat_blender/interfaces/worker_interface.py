from __future__ import annotations

import threading
from queue import Queue, Empty
from typing import Generic, TypeVar, Optional, Callable, Optional, ParamSpec, ClassVar
from functools import wraps
import time
from datetime import datetime

QT = TypeVar("QT")  # types elements of worker queue
P = ParamSpec("P")
R = TypeVar("R")

from ..infrastructure.mixins import MosplatLogClassMixin


def record_work_time(
    fn: Callable[P, R],
    start_callback: Optional[Callable[[float], None]] = None,
    end_callback: Optional[Callable[[float, float], None]] = None,
) -> Callable[P, R]:
    """
    a decorator / function wrapper that will track execution time
    and execute given callbacks.
    `start_time` and `end_time` are float values determined using `datetime.now().timestamp()

    Args:
        fn: Any function, whose exact signature will be preserved.
        start_callback:
            def start_callback(start_time: float) -> None: ...
        end_callback:
            def end_callback(end_time: float, time_elapsed: float) -> None: ...

    Returns:
        Wrapped function (with exact signature).
    """

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start_time = datetime.now().timestamp()
        start_counter = time.perf_counter()

        if start_callback is not None:
            start_callback(start_time)

        ret = fn(*args, **kwargs)

        end_time = datetime.now().timestamp()
        end_counter = time.perf_counter()
        time_elapsed = end_counter - start_counter

        if end_callback is not None:
            end_callback(end_time, time_elapsed)

        return ret

    return wrapper


class MosplatWorkerInterface(Generic[QT], MosplatLogClassMixin):
    """
    this is a clean abstraction for blender operators that require similar "worker" behavior in a separate thread

    features execution through threading (and start / end callbacks if necessary),
    a queue + polling mechanism for maintaining communication with main thread,
    and `threading.Event`-based cancellation.
    """

    def __init__(
        self,
        owner_name: str,
        worker_fn: Callable[[Queue[QT], threading.Event], None],
        *,
        use_start_work_callback: bool = True,
        start_work_callback: Optional[Callable[[float], None]] = None,
        use_end_work_callback: bool = True,
        end_work_callback: Optional[Callable[[float, float], None]] = None,
    ):
        # create queue and cancel event
        self._queue: Queue[QT] = Queue[QT]()
        self._cancel_event: threading.Event = threading.Event()

        # set up start work callback unless opt-ed out
        self._start_work_callback = (
            (start_work_callback or self._default_start_work_callback)
            if use_start_work_callback
            else None
        )
        self._end_work_callback = (
            (end_work_callback or self._default_end_work_callback)
            if use_end_work_callback
            else None
        )

        self._wrapped_fn = record_work_time(
            worker_fn, self._start_work_callback, self._end_work_callback
        )

        self._thread = threading.Thread(
            target=self._wrapped_fn,
            args=(self._queue, self._cancel_event),
            daemon=True,
        )
        self._id: Optional[int] = None
        self._owner_name: str = owner_name

    @property
    def identifier(self) -> str:
        return str(self._id) if self._id is not None else self._owner_name

    def start(self):
        self._thread.start()
        self._id = self._thread.native_id

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

    def _default_start_work_callback(self, start_time: float):
        start: str = datetime.fromtimestamp(start_time).strftime("%H:%M:%S")

        self.logger.info(
            f"Worker thread for '{self._owner_name}' started with ID '{self._id}' at time '{start}'."
        )

    def _default_end_work_callback(self, end_time: float, time_elapsed: float):
        end: str = datetime.fromtimestamp(end_time).strftime("%H:%M:%S")

        self.logger.info(
            f"Worker thread for '{self._owner_name}' with ID '{self._id}' ended at time '{end}'."
        )

        self.logger.info(f"Total time elapsed: '{time_elapsed:.2f}' seconds")
