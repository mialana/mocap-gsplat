from pathlib import Path
from typing import NamedTuple

from mosplat_blender.infrastructure.schemas import UnexpectedError
from mosplat_blender.interfaces import VGGTInterface
from mosplat_blender.operators.base_ot import MosplatOperatorBase


class Kwargs(NamedTuple):
    hf_id: str
    model_cache_dir: Path


class Mosplat_OT_initialize_model(
    MosplatOperatorBase[VGGTInterface.InitQueueTuple, Kwargs]
):
    @classmethod
    def _contexted_poll(cls, pkg):
        if VGGTInterface().model is not None:
            cls.poll_message_set("Model has already been initialized.")
            return False  # prevent re-initialization
        if pkg.props.progress_accessor.in_use:
            cls.poll_message_set("Wait until operators in-progress have completed.")
            return False
        return True

    def _queue_callback(self, pkg, event, next):
        status, msg, current, total = next
        props = pkg.props
        prefs = pkg.prefs
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
            return "RUNNING_MODAL"

        if status == "downloaded":
            self.cleanup_subprocess()
            self.launch_thread(
                pkg.context,
                twargs=Kwargs(
                    hf_id=prefs.vggt_hf_id,
                    model_cache_dir=prefs.cache_dir_vggt_,
                ),
            )

        return super()._queue_callback(pkg, event, next)

    def _contexted_execute(self, pkg):
        prefs = pkg.prefs

        self.logger.debug(f"Downloading model with subprocess.")

        self.launch_subprocess(
            pkg.context,
            pwargs=Kwargs(
                hf_id=prefs.vggt_hf_id,
                model_cache_dir=prefs.cache_dir_vggt_,
            ),
        )

        return "RUNNING_MODAL"

    def cleanup(self, pkg):
        self.wm(pkg.context).progress_end()  # stop progress
        progress = pkg.props.progress_accessor
        progress.in_use = False
        progress.current = -1
        progress.total = -1

        return super().cleanup(pkg)

    @staticmethod
    def _operator_subprocess(queue, cancel_event, *, pwargs):
        try:
            VGGTInterface.download_model(
                pwargs.hf_id, pwargs.model_cache_dir, queue, cancel_event
            )
        except UnexpectedError as e:
            queue.put(("error", str(e), -1, -1))

    @staticmethod
    def _operator_thread(queue, cancel_event, *, twargs):
        try:
            VGGTInterface().initialize_model(
                twargs.hf_id,
                twargs.model_cache_dir,
            )
            queue.put(("done", "Model initialized.", -1, -1))
        except Exception as e:
            queue.put(("error", str(e), -1, -1))


def process_entrypoint(*args, **kwargs):
    Mosplat_OT_initialize_model._operator_subprocess(*args, **kwargs)
