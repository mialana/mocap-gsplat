from typing import NamedTuple, Tuple

from interfaces import VGGTInterface
from operators.base_ot import MosplatOperatorBase


class ThreadKwargs(NamedTuple):
    pass


class Mosplat_OT_run_inference(MosplatOperatorBase[Tuple[str, str], ThreadKwargs]):
    @classmethod
    def _contexted_poll(cls, pkg):
        # if VGGTInterface().model is None:
        #     cls._poll_error_msg_list.append("Model must be initialized.")
        if not pkg.props.was_frame_range_extracted:
            cls._poll_error_msg_list.append("Frame range must be extracted.")

        return len(cls._poll_error_msg_list) == 0

    def _queue_callback(self, pkg, event, next):

        return super()._queue_callback(pkg, event, next)

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
        import time

        time.sleep(4)

        queue.put(("done", "we are done"))


def process_entrypoint(*args, **kwargs):
    Mosplat_OT_run_inference._operator_subprocess(*args, **kwargs)
