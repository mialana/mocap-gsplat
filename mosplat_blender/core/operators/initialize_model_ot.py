import bpy

from pathlib import Path
import threading
from queue import Queue
from typing import Tuple
import subprocess
import json
import time
import sys
import os

from ...infrastructure.macros import tuple_matches_type_tuple
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


class Mosplat_OT_initialize_model(MosplatOperatorBase[Tuple[str, int, int, str]]):
    bl_idname = OperatorIDEnum.INITIALIZE_MODEL
    bl_description = (
        f"Install VGGT model weights from Hugging Face or load from cache if available."
    )

    @classmethod
    def contexted_poll(cls, context, prefs, props) -> bool:
        from ...interfaces import MosplatVGGTInterface

        if MosplatVGGTInterface._model is not None:
            cls.poll_message_set("Model has already been initialized.")
            return False  # prevent re-initialization
        return True

    def queue_callback(self, context, event, next) -> OptionalOperatorReturnItemsSet:
        status, current, total, msg = next
        if status == "progress":
            self.logger().info(f"{current >> 20 }MiB / {total >> 20 }MiB")
        elif status == "error":
            self.logger().error(msg)
            return {"CANCELLED"}
        else:
            fmt = f"{status.upper()} - {msg}"
            (
                self.logger().warning(fmt)
                if status == "warning"
                else self.logger().info(fmt)
            )
            if status == "done":
                self.cleanup(context)
                return {"FINISHED"}

    def contexted_execute(self, context) -> OperatorReturnItemsSet:
        prefs = self.prefs

        if not is_path_accessible(DOWNLOAD_HF_WITH_PROGRESS_SCRIPT_PATH):
            self.logger().error(
                f"'{DOWNLOAD_HF_WITH_PROGRESS_SCRIPT_PATH}' script file not found at expected location."
            )
            return {"CANCELLED"}

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
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            preexec_fn=getattr(os, "setsid", None),
        )

        self.logger().debug(f"PID: {self._proc.pid}")

        initialize_model_thread(
            self,
            proc=self._proc,
            hf_id=prefs.vggt_hf_id,
            model_cache_dir=prefs.vggt_model_dir,
        )

        return {"RUNNING_MODAL"}


@worker_fn_auto
def initialize_model_thread(
    queue: Queue[Tuple[str, int, int, str]],
    cancel_event: threading.Event,
    *,
    proc: subprocess.Popen,
    hf_id: str,
    model_cache_dir: Path,
):
    try:
        _wait_and_update_queue_loop(proc, queue, cancel_event)
    except SafeError:
        pass

    from ...interfaces import MosplatVGGTInterface

    try:
        MosplatVGGTInterface.initialize_model(hf_id, model_cache_dir, cancel_event)

        queue.put(("done", -1, -1, "Successfully downloaded & initialized VGGT model!"))
    except UnexpectedError as e:
        queue.put(("error", -1, -1, str(e)))


def KILL(pid: int):
    if sys.platform != "win32":
        os.killpg(proc.pid, signal.SIGKILL)
    else:
        import psutil

        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except psutil.NoSuchProcess:
            pass


def _wait_and_update_queue_loop(
    proc: subprocess.Popen,
    queue: Queue[Tuple[str, int, int, str]],
    cancel_event: threading.Event,
):
    TYPE_TUPLE = (str, int, int, str)  # for runtime checking

    def check():
        if cancel_event.is_set():
            KILL(proc.pid)
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
                except json.JSONDecodeError:
                    pass
        if proc.stderr is not None:
            for line in proc.stderr:
                check()
                queue.put(("warning", -1, -1, line))

        if proc.poll() is not None:
            return

        time.sleep(_TIMER_INTERVAL_)
