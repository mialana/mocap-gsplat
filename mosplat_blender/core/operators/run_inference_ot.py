import ast
import os
import subprocess
from functools import partial
from pathlib import Path
from queue import Queue
from typing import Final, NamedTuple, Tuple

import bpy

from ...infrastructure.constants import (
    _TIMEOUT_INTERVAL_,
    DOWNLOAD_HF_WITH_PROGRESS_SCRIPT,
)
from ...infrastructure.macros import kill_subprocess_cross_platform, try_access_path
from ...infrastructure.schemas import UnexpectedError, UserFacingError
from .base_ot import MosplatOperatorBase


class ThreadKwargs(NamedTuple):
    pass


class Mosplat_OT_run_inference(
    MosplatOperatorBase[Tuple[str, int, int, str], ThreadKwargs]
):
    @classmethod
    def _contexted_poll(cls, pkg):
        from ...interfaces import MosplatVGGTInterface

        if MosplatVGGTInterface().model is None:
            cls.poll_message_set("Model must be initialized.")
            return False  # prevent without model initialization
        return True

    def _queue_callback(self, pkg, event, next):

        return "RUNNING_MODAL"

    def _contexted_invoke(self, pkg, event):
        return self.execute_with_package(pkg)

    def _contexted_execute(self, pkg):
        prefs = pkg.prefs

        self.launch_thread(
            pkg.context,
            twargs=ThreadKwargs(),
        )

        return "RUNNING_MODAL"

    @staticmethod
    def _operator_thread(queue, cancel_event, *, twargs):
        pass
