from __future__ import annotations

import multiprocessing as mp
import multiprocessing.context as mp_ctx
import multiprocessing.synchronize as mp_sync
import os
import queue
import threading
import time
from datetime import datetime
from typing import Callable, Generic, Optional, TypeAlias, TypeVar

QT = TypeVar("QT", bound=tuple)  # types elements of worker queue


from mosplat_blender.infrastructure.constants import _TIMEOUT_LAZY_
from mosplat_blender.infrastructure.decorators import record_work_time
from mosplat_blender.infrastructure.macros import kill_subprocess_cross_platform
from mosplat_blender.infrastructure.mixins import LogClassMixin
from mosplat_blender.infrastructure.schemas import EnvVariableEnum

StartWorkCallback: TypeAlias = Optional[Callable[[float], None]]
EndWorkCallback: TypeAlias = Optional[Callable[[float, float], None]]


class SubprocessWorkerInterface(Generic[QT], LogClassMixin):
    """
    this is a clean abstraction for blender operators that require similar "worker" behavior in a subprocess

    features execution through `multiprocessing` (and start / end callbacks if necessary),
    a queue + polling mechanism for maintaining communication with main process,
    event-based cancellation, and a watcher thread to catch the subprocess' end.
    """

    def __init__(
        self, owner_name: str, worker_fn: Callable[[mp.Queue[QT], mp_sync.Event], None]
    ):
        self._ctx: mp_ctx.SpawnContext = mp.get_context("spawn")

        # create queue for process to send data back to main process
        self._queue: mp.Queue[QT] = self._ctx.Queue()
        # create event for main process to signal subprocess to cancel itself
        self._cancel_event: mp_sync.Event = self._ctx.Event()

        self._process: mp_ctx.SpawnProcess = self._ctx.Process(
            target=worker_fn,
            args=(self._queue, self._cancel_event),
            daemon=True,
        )
        self._pid: Optional[int] = None
        self._owner_name: str = owner_name

        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.time_elapsed: Optional[float] = None
        self._start_counter: Optional[float] = None
        self._end_counter: Optional[float]

    @property
    def identifier(self) -> str:
        return str(self._pid) if self._pid is not None else self._owner_name

    def start(self):
        os.environ.setdefault(EnvVariableEnum.SUBPROCESS_FLAG, "1")
        try:
            self._process.start()
        finally:  # ensure flag never stays in env
            os.environ.pop(EnvVariableEnum.SUBPROCESS_FLAG)
        self._pid = self._process.pid

        self._start_timer()  # start timer after process begins so that pid exists

    def cancel(self):
        self._cancel_event.set()

        def reaper_thread():
            # give the process an ample of time to finish, then force termination
            self._process.join(timeout=_TIMEOUT_LAZY_)
            self.force_terminate()

        threading.Thread(target=reaper_thread, daemon=True).start()

    def was_cancelled(self):
        return not self._cancel_event.is_set()

    def is_alive(self):
        return self._process.is_alive()

    def cleanup(self):
        self.cancel()

        # drain queue
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self.logger.debug(f"'{self.identifier}' clean.")

    def dequeue(self) -> Optional[QT]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def force_terminate(self):
        if self._pid:
            kill_subprocess_cross_platform(self._pid)

    def _start_timer(self):
        self.start_time = datetime.now().timestamp()
        start: str = datetime.fromtimestamp(self.start_time).strftime("%H:%M:%S")

        self.logger.info(
            f"Worker process for '{self._owner_name}' started with ID '{self.identifier}' at time '{start}'."
        )
        self._start_counter = time.perf_counter()

        def watcher_thread():
            # join the process and wait for end
            self._process.join()
            self._end_timer()

        threading.Thread(target=watcher_thread, daemon=True).start()

    def _end_timer(self):
        if not self._start_counter:
            self.logger.warning("No start counter available for worker")
            return
        self._end_counter = time.perf_counter()
        self.end_time = datetime.now().timestamp()

        end: str = datetime.fromtimestamp(self.end_time).strftime("%H:%M:%S")

        self.logger.info(
            f"Worker process for '{self._owner_name}' with ID '{self.identifier}' ended at time '{end}'."
        )

        self.time_elapsed = self._end_counter - self._start_counter

        self.logger.info(f"Total time elapsed: '{self.time_elapsed:.2f}' seconds")


class ThreadWorkerInterface(Generic[QT], LogClassMixin):
    """
    this is a clean abstraction for blender operators that require similar "worker" behavior in a separate thread

    features execution through threading (and start / end callbacks if necessary),
    a queue + polling mechanism for maintaining communication with main thread,
    and `threading.Event`-based cancellation.
    """

    def __init__(
        self,
        owner_name: str,
        worker_fn: Callable[[queue.Queue[QT], threading.Event], None],
    ):
        # create queue and cancel event
        self._queue: queue.Queue[QT] = queue.Queue[QT]()
        self._cancel_event: threading.Event = threading.Event()

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
            except queue.Empty:
                break
        self.logger.debug(f"'{self.identifier}' clean.")

    def dequeue(self) -> Optional[QT]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def _start_work_callback(self, start_time: float):
        start: str = datetime.fromtimestamp(start_time).strftime("%H:%M:%S")

        self.logger.info(
            f"Worker thread for '{self._owner_name}' started with ID '{self._id}' at time '{start}'."
        )

    def _end_work_callback(self, end_time: float, time_elapsed: float):
        end: str = datetime.fromtimestamp(end_time).strftime("%H:%M:%S")

        self.logger.info(
            f"Worker thread for '{self._owner_name}' with ID '{self._id}' ended at time '{end}'."
        )

        self.logger.info(f"Total time elapsed: '{time_elapsed:.2f}' seconds")
