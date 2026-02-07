import ast
import os
import subprocess
from functools import partial
from pathlib import Path
from typing import Final, NamedTuple, Tuple

from infrastructure.constants import (
    _TIMEOUT_LAZY_,
    DOWNLOAD_HF_WITH_PROGRESS_SCRIPT,
)
from infrastructure.macros import (
    kill_subprocess_cross_platform,
    try_access_path,
    tuple_type_matches_known_tuple_type,
)
from infrastructure.schemas import UnexpectedError, UserFacingError
from interfaces import MosplatVGGTInterface
from operators.base_ot import MosplatOperatorBase

QUEUE_DEFAULT_TUPLE: Final = ("", 0, 0, "")  # for runtime check against unknown tuples


class ProcessKwargs(NamedTuple):
    hf_id: str
    model_cache_dir: Path


class Mosplat_OT_initialize_model(
    MosplatOperatorBase[Tuple[str, int, int, str], ProcessKwargs]
):
    @classmethod
    def _contexted_poll(cls, pkg):
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
        fmt = f"QUEUE {status.upper()} - {msg}"
        match status:
            case "progress":
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
            case "error":
                self.logger.error(fmt)
                return "FINISHED"  # return finished as blender data has been modified
            case "warning":
                self.logger.warning(fmt)
            case "done":
                self.logger.info(fmt)
            case _:
                self.logger.debug(fmt)

        match status:
            case "error" | "done":
                return "FINISHED"
            case _:
                return "RUNNING_MODAL"

    def _contexted_execute(self, pkg):
        prefs = pkg.prefs

        self.logger.debug(f"Downloading model with subprocess.")

        self.launch_subprocess(
            pkg.context,
            pwargs=ProcessKwargs(
                hf_id=prefs.vggt_hf_id,
                model_cache_dir=prefs.vggt_model_dir,
            ),
        )

        return "RUNNING_MODAL"

    def _contexted_modal(self, pkg, event):
        # check cancellation on timer callback, kill subprocess if needed
        if self.worker and not self.worker.is_alive():
            self.worker.force_terminate()
        return super()._contexted_modal(pkg, event)

    def cleanup(self, pkg):
        from bpy import app

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
        if self.worker and self.worker._pid:
            app.timers.register(
                partial(kill_subprocess_cross_platform, self.worker._pid),
                first_interval=_TIMEOUT_LAZY_,  # no orphaned processes here!
            )
        return super().cleanup(pkg)

    @staticmethod
    def _operator_subprocess(queue, cancel_event, *, pwargs):
        try:
            MosplatVGGTInterface().initialize_model(
                pwargs.hf_id, pwargs.model_cache_dir, queue, cancel_event
            )

            queue.put(("done", -1, -1, "Successfully downloaded & init VGGT model!"))
        except UnexpectedError as e:
            queue.put(("error", -1, -1, str(e)))


def process_entrypoint(*args, **kwargs):
    Mosplat_OT_initialize_model._operator_subprocess(*args, **kwargs)
