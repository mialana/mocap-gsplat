import bpy

from pathlib import Path
import threading
from queue import Queue
from typing import Tuple
import subprocess
import json
import time
import os
from dataclasses import dataclass

from ...infrastructure.macros import (
    tuple_matches_type_tuple,
    kill_subprocess_cross_platform,
)
from ...infrastructure.schemas import OperatorIDEnum, UnexpectedError, SafeError
from ...infrastructure.decorators import worker_fn_auto
from ...infrastructure.constants import (
    DOWNLOAD_HF_WITH_PROGRESS_SCRIPT_PATH,
    _TIMER_INTERVAL_,
)
from ...infrastructure.macros import is_path_accessible

from .base_ot import (
    MosplatOperatorBase,
    OperatorReturnItemsSet,
    OptionalOperatorReturnItemsSet,
)


@dataclass(frozen=True)
class ThreadKwargs:
    proc: subprocess.Popen
    hf_id: str
    model_cache_dir: Path


class Mosplat_OT_initialize_model(
    MosplatOperatorBase[Tuple[str, int, int, str], ThreadKwargs]
):
    bl_idname = OperatorIDEnum.INITIALIZE_MODEL
    bl_description = (
        f"Install VGGT model weights from Hugging Face or load from cache if available."
    )

    @classmethod
    def contexted_poll(cls, context, prefs, props) -> bool:
        from ...interfaces import MosplatVGGTInterface

        if MosplatVGGTInterface.model is not None:
            cls.poll_message_set("Model has already been initialized.")
            return False  # prevent re-initialization
        if props.progress_in_use:
            cls.poll_message_set("Wait until operators in-progress have completed.")
            return False
        return True

    def queue_callback(self, context, event, next) -> OptionalOperatorReturnItemsSet:
        status, current, total, msg = next
        props = self.props
        wm = self.wm
        if status == "progress":
            if not props.progress_in_use and total > 0:
                wm.progress_begin(current, total)  # start progress bar if needed
                props.progress_in_use = True  # mark usage of global progress props
                props.operator_progress_total = total
            if total != self.props.operator_progress_total:  # in case total changes
                wm.progress_end()
                wm.progress_begin(current, total)
                props.operator_progress_total = total

            self.wm.progress_update(current)
            self.props.operator_progress_current = current
        else:
            fmt = f"QUEUE {status.upper()} - {msg}"
            if status == "error":
                self.cleanup(context)
                self.logger().error(fmt)
                return {"CANCELLED"}
            elif status == "warning":
                self.logger().warning(fmt)
            elif status == "debug":
                self.logger().debug(fmt)
            else:
                self.logger().info(fmt)
            if status == "done":
                self.cleanup(context)
                return {"FINISHED"}

    def contexted_execute(self, context) -> OperatorReturnItemsSet:
        if not is_path_accessible(DOWNLOAD_HF_WITH_PROGRESS_SCRIPT_PATH):
            self.logger().error(
                f"'{DOWNLOAD_HF_WITH_PROGRESS_SCRIPT_PATH}' script file not found at expected location."
            )
            return {"CANCELLED"}

        prefs = self.prefs

        # start the model download from a separate subprocess
        self._proc: subprocess.Popen = subprocess.Popen(
            [
                bpy.app.binary_path,
                "--factory-startup",
                "-b",
                "--python",
                DOWNLOAD_HF_WITH_PROGRESS_SCRIPT_PATH,
                "--",
                prefs.vggt_hf_id,
                str(prefs.vggt_model_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            preexec_fn=getattr(os, "setsid", None),  # TODO: cross-platform check
        )

        self.logger().info(f"Downloading model from subprocess. PID: {self._proc.pid}")

        self.operator_thread(
            self,
            _kwargs=ThreadKwargs(
                proc=self._proc,
                hf_id=prefs.vggt_hf_id,
                model_cache_dir=prefs.vggt_model_dir,
            ),
        )

        return {"RUNNING_MODAL"}

    @staticmethod
    @worker_fn_auto
    def operator_thread(queue, cancel_event, *, _kwargs):
        try:
            _wait_and_update_queue_loop(_kwargs.proc, queue, cancel_event)
        except SafeError:
            # put on queue to stop modal just in case, though it probably will not be read
            queue.put(
                ("done", -1, -1, "Model init did not finish due to cancellation.")
            )
            return  # cancel event was set

        from ...interfaces import MosplatVGGTInterface

        try:
            MosplatVGGTInterface.initialize_model(
                _kwargs.hf_id, _kwargs.model_cache_dir, cancel_event
            )

            queue.put(
                ("done", -1, -1, "Successfully downloaded & initialized VGGT model!")
            )
        except UnexpectedError as e:
            queue.put(("error", -1, -1, str(e)))


def _wait_and_update_queue_loop(
    proc: subprocess.Popen,
    queue: Queue[Tuple[str, int, int, str]],
    cancel_event: threading.Event,
):
    TYPE_TUPLE = (str, int, int, str)  # for runtime checking

    def check():
        if cancel_event.is_set():
            kill_subprocess_cross_platform(proc.pid)
            raise SafeError

    while True:
        check()

        if proc.stdout is not None:
            for line in proc.stdout:
                check()
                try:
                    converted = tuple(dict(json.loads(line)).values())
                    if tuple_matches_type_tuple(converted, TYPE_TUPLE):
                        queue.put(converted)  # good to go
                        continue
                except json.JSONDecodeError:
                    pass

                if isinstance(line, str) and (stripped := line.strip()):
                    # place other output in queue under msg
                    queue.put(("debug", -1, -1, stripped))

        if proc.poll() is not None:
            return

        time.sleep(_TIMER_INTERVAL_)
