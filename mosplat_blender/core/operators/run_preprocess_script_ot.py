from __future__ import annotations

from typing import List, NamedTuple
from dataclasses import dataclass
from pathlib import Path

from .base_ot import (
    MosplatOperatorBase,
    OperatorReturnItemsSet,
    OptionalOperatorReturnItemsSet,
)

from ..handlers import restore_dataset_from_json

from ...infrastructure.schemas import OperatorIDEnum, UserFacingError, MediaIODataset
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
    def contexted_poll(cls, context, prefs, props) -> bool:
        if not props.dataset_accessor.is_valid_media_directory:
            cls._poll_error_msg_list.append(
                "Ensure that frame count, width, and height of all media files within current media directory match."
            )
        else:
            cls._poll_error_msg_list.extend(props.frame_range_err_list(prefs))

        return len(cls._poll_error_msg_list) == 0

    def contexted_invoke(self, context, event) -> OperatorReturnItemsSet:
        prefs = self.prefs
        props = self.props
        try:
            restore_dataset_from_json(props, prefs)  # try to restore from local JSON

            self._preprocess_script = prefs.preprocess_media_script_filepath
            self._npy_filepaths: List[Path] = props.frame_range_npy_filepaths(
                prefs, RAW_FRAME_DIRNAME
            )

            return self.execute(context)
        except UserFacingError as e:
            self.logger.error(str(e))
            return {"CANCELLED"}

    def contexted_execute(self, context) -> OperatorReturnItemsSet:

        self.operator_thread(
            self,
            _kwargs=ThreadKwargs(
                preprocess_script=self._preprocess_script,
                raw_npy_filepaths=self._npy_filepaths,
                dataset_as_dc=self.dataset_as_dc,
            ),
        )

        return {"RUNNING_MODAL"}

    def queue_callback(self, context, event, next) -> OptionalOperatorReturnItemsSet:
        if next == "done":
            self.cleanup(context)  # write props (as dataclass) to JSON
            return

        if next != "update":  # if sent an error message via queue
            self.logger.warning(next)

        # sync props regardless as the updated dataclass is still valid
        self.props.dataset_accessor.from_dataclass(self.dataset_as_dc)

    @staticmethod
    @worker_fn_auto
    def operator_thread(queue, cancel_event, *, _kwargs):
        preprocess_script = _kwargs.preprocess_script

        for fp in _kwargs.raw_npy_filepaths:
            pass
