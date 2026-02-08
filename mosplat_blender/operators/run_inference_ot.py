from typing import NamedTuple, Tuple

from interfaces import VGGTInterface
from operators.base_ot import MosplatOperatorBase


class ProcessKwargs(NamedTuple):
    pass


class Mosplat_OT_run_inference(
    MosplatOperatorBase[Tuple[str, int, int, str], ProcessKwargs]
):
    @classmethod
    def _contexted_poll(cls, pkg):

        if VGGTInterface().model is None:
            cls.poll_message_set("Model must be initialized.")
            return False  # prevent without model initialization
        return True

    def _queue_callback(self, pkg, event, next):

        return "RUNNING_MODAL"

    def _contexted_invoke(self, pkg, event):
        return self.execute_with_package(pkg)

    def _contexted_execute(self, pkg):
        prefs = pkg.prefs

        self.launch_subprocess(
            pkg.context,
            pwargs=ProcessKwargs(),
        )

        return "RUNNING_MODAL"

    @staticmethod
    def _operator_subprocess(queue, cancel_event, *, pwargs):
        pass


def process_entrypoint(*args, **kwargs):
    Mosplat_OT_run_inference._operator_subprocess(*args, **kwargs)
