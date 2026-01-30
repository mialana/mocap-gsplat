import bpy

from pathlib import Path
from queue import Queue
from typing import Tuple, NamedTuple, Final
import subprocess
import os
import ast
from functools import partial

from ...infrastructure.macros import (
    tuple_type_matches_known_tuple_type,
    kill_subprocess_cross_platform,
)
from ...infrastructure.schemas import OperatorIDEnum, UnexpectedError
from ...infrastructure.decorators import worker_fn_auto
from ...infrastructure.constants import (
    DOWNLOAD_HF_WITH_PROGRESS_SCRIPT,
    _TIMEOUT_INTERVAL_,
)
from ...infrastructure.macros import is_path_accessible

from .base_ot import MosplatOperatorBase

QUEUE_DEFAULT_TUPLE: Final = ("", 0, 0, "")  # for runtime check against unknown tuples


class ThreadKwargs(NamedTuple):
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
    def contexted_poll(cls, pkg):
        from ...interfaces import MosplatVGGTInterface

        if MosplatVGGTInterface.model is not None:
            cls.poll_message_set("Model has already been initialized.")
            return False  # prevent re-initialization
        if pkg.props.progress_in_use:
            cls.poll_message_set("Wait until operators in-progress have completed.")
            return False
        return True

    def queue_callback(self, pkg, event, next):
        status, current, total, msg = next
        props = pkg.props
        wm = self.wm(pkg.context)
        if status == "progress":
            if not props.progress_in_use and total > 0:
                wm.progress_begin(current, total)  # start progress bar if needed
                props.progress_in_use = True  # mark usage of global progress props
                props.operator_progress_total = total
            if total != props.operator_progress_total:  # in case total changes
                wm.progress_end()
                wm.progress_begin(current, total)
                props.operator_progress_total = total

            wm.progress_update(current)
            props.operator_progress_current = current
            self.logger.debug(f"{current} / {total}")
        else:
            fmt = f"QUEUE {status.upper()} - {msg}"
            if status == "error":
                self.logger.error(fmt)
                return "CANCELLED"  # finish

            elif status == "warning":
                self.logger.warning(fmt)
            elif status == "debug":
                self.logger.debug(fmt)
            else:
                self.logger.info(fmt)
            if status == "done":
                return "FINISHED"  # finish

    def contexted_execute(self, pkg):
        if not is_path_accessible(DOWNLOAD_HF_WITH_PROGRESS_SCRIPT):
            self.logger.error(
                f"'{DOWNLOAD_HF_WITH_PROGRESS_SCRIPT}' script file not found at expected location."
            )

            return "CANCELLED"

        prefs = pkg.prefs

        # start the model download from a separate subprocess
        self._proc: subprocess.Popen = subprocess.Popen(
            [
                bpy.app.binary_path,
                "--factory-startup",
                "-b",
                "--python",
                DOWNLOAD_HF_WITH_PROGRESS_SCRIPT,
                "--",
                prefs.vggt_hf_id,
                str(prefs.vggt_model_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True,
            encoding="utf-8",
            preexec_fn=getattr(os, "setsid", None),  # TODO: cross-platform check
        )

        self.logger.info(f"Downloading model from subprocess. PID: {self._proc.pid}")

        self.operator_thread(
            self,
            pkg.context,
            _kwargs=ThreadKwargs(
                proc=self._proc,
                hf_id=prefs.vggt_hf_id,
                model_cache_dir=prefs.vggt_model_dir,
            ),
        )

        return "RUNNING_MODAL"

    def contexted_modal(self, pkg, event):
        # check cancellation on timer callback, kill subprocess if needed
        if self.worker and self.worker.was_cancelled():
            kill_subprocess_cross_platform(self._proc.pid)
        super().contexted_modal(pkg, event)

    def cleanup(self, pkg):
        self.wm(pkg.context).progress_end()  # stop progress
        pkg.props.progress_in_use = False
        pkg.props.operator_progress_current = -1
        pkg.props.operator_progress_total = -1

        """
        register a timer that will kill the subprocess with the saved pid
        (operator itself will leave the scope before then)
        this ensures that cancellation that never reaches the next modal call 
        still cleans up the subprocess
        """
        if hasattr(self, "_proc"):
            bpy.app.timers.register(
                partial(kill_subprocess_cross_platform, self._proc.pid),
                first_interval=_TIMEOUT_INTERVAL_,  # no orphaned processes here!
            )
        return super().cleanup(pkg)

    @staticmethod
    @worker_fn_auto
    def operator_thread(queue, cancel_event, *, _kwargs):
        try:
            _wait_and_update_queue_loop(_kwargs.proc, queue)
        except (subprocess.CalledProcessError, UnexpectedError) as e:
            # put on queue to stop modal just in case, though it probably will not be read
            queue.put(("error", -1, -1, str(e)))
            return  # do not continue
        if cancel_event.is_set():
            return  # do not continue

        from ...interfaces import MosplatVGGTInterface

        try:
            MosplatVGGTInterface.initialize_model(
                _kwargs.hf_id, _kwargs.model_cache_dir, cancel_event
            )

            queue.put(("done", -1, -1, "Successfully downloaded & init VGGT model!"))
        except UnexpectedError as e:
            queue.put(("error", -1, -1, str(e)))


def _wait_and_update_queue_loop(
    proc: subprocess.Popen, queue: Queue[Tuple[str, int, int, str]]
):
    if proc.stdout is None:
        raise UnexpectedError("Process stdout pointer is not available.")

    for line in proc.stdout:  # continues until EOF has been reached
        try:
            converted = ast.literal_eval(line)
            if isinstance(converted, tuple) and tuple_type_matches_known_tuple_type(
                converted, QUEUE_DEFAULT_TUPLE
            ):
                queue.put(converted)  # good to go
                continue
        except (ValueError, SyntaxError, TypeError):
            pass  # ignore errors raised from trying to evaluate the line as a tuple
        if isinstance(line, str) and (stripped := line.strip()):
            # place other output in queue under msg
            queue.put(("debug", -1, -1, stripped))

    # avoid concurrency issues with `poll` when stdout is released by process
    ret: int = proc.wait()
    if ret != 0:
        e = subprocess.CalledProcessError(ret, proc.args)
        e.add_note("Note: ignore this if operator was deliberately cancelled.")
        raise e

    return  # proc existed successfully
