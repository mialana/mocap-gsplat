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
            cls._poll_error_msg_list.append("Model must be initialized.")
        if not pkg.props.was_frame_range_extracted:
            cls._poll_error_msg_list.append("Frame range must be extracted.")

        return len(cls._poll_error_msg_list) == 0

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
