import ast
import os
import subprocess
from functools import partial
from pathlib import Path
from queue import Queue
from typing import Final, NamedTuple, Tuple

import bpy

from core.operators.base_ot import MosplatOperatorBase
from infrastructure.constants import (
    _TIMEOUT_INTERVAL_,
    DOWNLOAD_HF_WITH_PROGRESS_SCRIPT,
)
from infrastructure.macros import (
    kill_subprocess_cross_platform,
    try_access_path,
    tuple_type_matches_known_tuple_type,
)
from infrastructure.schemas import UnexpectedError, UserFacingError

QUEUE_DEFAULT_TUPLE: Final = ("", 0, 0, "")  # for runtime check against unknown tuples


class ThreadKwargs(NamedTuple):
    proc: subprocess.Popen
    hf_id: str
    model_cache_dir: Path


class Mosplat_OT_initialize_model(
    MosplatOperatorBase[Tuple[str, int, int, str], ThreadKwargs]
):
    @classmethod
    def _contexted_poll(cls, pkg):
        from ...interfaces import MosplatVGGTInterface

        if MosplatVGGTInterface().model is not None:
            cls.poll_message_set("Model has already been initialized.")
            return False  # prevent re-initialization
        if pkg.props.progress_accessor.in_use:
            cls.poll_message_set("Wait until operators in-progress have completed.")
            return False
        return True

    def _queue_callback(self, pkg, event, next):
        status, current, total, msg = next
        props = pkg.props
        wm = self.wm(pkg.context)
        progress = props.progress_accessor
        if status == "progress":
            if not progress.in_use and total > 0:
                wm.progress_begin(current, total)  # start progress bar if needed
                progress.in_use = True  # mark usage of global progress props
                progress.total = total
            if total != progress.total:  # in case total changes
                wm.progress_end()
                wm.progress_begin(current, total)
                progress.total = total

            wm.progress_update(current)
            progress.current = current
            self.logger.debug(f"{current} / {total}")
        else:
            fmt = f"QUEUE {status.upper()} - {msg}"
            if status == "error":
                self.logger.error(fmt)
                return "FINISHED"  # return finished as blender data has been modified

            elif status == "warning":
                self.logger.warning(fmt)
            elif status == "debug":
                self.logger.debug(fmt)
            else:
                self.logger.info(fmt)
            if status == "done":
                return "FINISHED"  # finish
        return "RUNNING_MODAL"

    def _contexted_invoke(self, pkg, event):
        script = DOWNLOAD_HF_WITH_PROGRESS_SCRIPT

        try:
            try_access_path(DOWNLOAD_HF_WITH_PROGRESS_SCRIPT)
        except (FileNotFoundError, OSError, PermissionError) as e:
            raise UserFacingError(
                f"'{script} not accessible at runtime."
                "This is a production-level script and should not be moved.",
                e,
            ) from e
        else:
            self._subprocess_script = script
            return self.execute_with_package(pkg)

    def _contexted_execute(self, pkg):
        prefs = pkg.prefs

        # start the model download from a separate subprocess
        self._proc: subprocess.Popen = subprocess.Popen(
            [
                bpy.app.binary_path,
                "--factory-startup",
                "-b",
                "--python",
                self._subprocess_script,
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

        self.launch_thread(
            pkg.context,
            twargs=ThreadKwargs(
                proc=self._proc,
                hf_id=prefs.vggt_hf_id,
                model_cache_dir=prefs.vggt_model_dir,
            ),
        )

        return "RUNNING_MODAL"

    def _contexted_modal(self, pkg, event):
        # check cancellation on timer callback, kill subprocess if needed
        if self.worker and self.worker.was_cancelled():
            kill_subprocess_cross_platform(self._proc.pid)
        return super()._contexted_modal(pkg, event)

    def cleanup(self, pkg):
        self.wm(pkg.context).progress_end()  # stop progress
        progress = pkg.props.progress_accessor
        progress.in_use = False
        progress.current = -1
        progress.total = -1

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
    def _operator_thread(queue, cancel_event, *, twargs):
        try:
            _wait_and_update_queue_loop(twargs.proc, queue)
        except (subprocess.CalledProcessError, UnexpectedError) as e:
            # put on queue to stop modal just in case, though it probably will not be read
            queue.put(("error", -1, -1, str(e)))
            return  # do not continue
        if cancel_event.is_set():
            return  # do not continue

        from ...interfaces import MosplatVGGTInterface

        try:
            MosplatVGGTInterface().initialize_model(
                twargs.hf_id, twargs.model_cache_dir, cancel_event
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
        e.add_note(
            "NOTE: Occurred while downloading Hugging Face model via subprocess."
        )
        raise e

    return  # proc existed successfully
