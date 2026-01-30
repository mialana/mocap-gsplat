from __future__ import annotations

from typing import List, NamedTuple
from pathlib import Path

from .base_ot import MosplatOperatorBase

from ..handlers import restore_dataset_from_json

from ...infrastructure.schemas import OperatorIDEnum, MediaIODataset
from ...infrastructure.decorators import worker_fn_auto
from ...infrastructure.constants import RAW_FRAME_DIRNAME


class ThreadKwargs(NamedTuple):
    preprocess_script: Path
    raw_npy_filepaths: List[Path]
    dataset_as_dc: MediaIODataset


class Mosplat_OT_run_preprocess_script(
    MosplatOperatorBase[str, ThreadKwargs],
):
    bl_idname = OperatorIDEnum.RUN_PREPROCESS_SCRIPT
    bl_description = "Run current preprocess script on current frame range."

    @classmethod
    def contexted_poll(cls, pkg) -> bool:
        props = pkg.props
        cls._poll_error_msg_list.extend(props.is_valid_media_directory_poll_result)
        cls._poll_error_msg_list.extend(props.frame_range_poll_result(pkg.prefs))

        return len(cls._poll_error_msg_list) == 0

    def contexted_invoke(self, pkg, event):
        prefs = pkg.prefs
        props = pkg.props

        restore_dataset_from_json(props, prefs)  # try to restore from local JSON

        self._preprocess_script = prefs.preprocess_media_script_filepath
        self._npy_filepaths: List[Path] = props.frame_range_npy_filepaths(
            prefs, RAW_FRAME_DIRNAME
        )

        return self.execute(pkg)

    def contexted_execute(self, pkg):
        self.operator_thread(
            self,
            pkg.context,
            _kwargs=ThreadKwargs(
                preprocess_script=self._preprocess_script,
                raw_npy_filepaths=self._npy_filepaths,
                dataset_as_dc=self.data,
            ),
        )

        return "RUNNING_MODAL"

    def queue_callback(self, pkg, event, next):
        if next == "done":
            return "FINISHED"

        if next != "update":  # if sent an error message via queue
            self.logger.warning(next)

        # sync props regardless as the updated dataclass is still valid
        pkg.props.dataset_accessor.from_dataclass(self.data)

    @staticmethod
    @worker_fn_auto
    def operator_thread(queue, cancel_event, *, _kwargs):
        preprocess_script = _kwargs.preprocess_script

        for fp in _kwargs.raw_npy_filepaths:
            pass
