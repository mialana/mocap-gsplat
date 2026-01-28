from __future__ import annotations

from typing import Tuple, List
from pathlib import Path
from dataclasses import dataclass

from .base_ot import (
    MosplatOperatorBase,
    OperatorReturnItemsSet,
    OptionalOperatorReturnItemsSet,
)
from ..handlers import restore_dataset_from_json

from ...infrastructure.schemas import (
    OperatorIDEnum,
    UserFacingError,
    MediaIODataset,
    MediaFileStatus,
)
from ...infrastructure.decorators import worker_fn_auto


@dataclass(frozen=True)
class ThreadKwargs:
    updated_media_files: List[Path]
    dataset_as_dc: MediaIODataset


class Mosplat_OT_validate_media_file_statuses(
    MosplatOperatorBase[Tuple[bool, str], ThreadKwargs]
):
    bl_idname = OperatorIDEnum.VALIDATE_MEDIA_FILE_STATUSES
    bl_description = "Check frame count, width, and height of all media files found in current media directory."

    def contexted_invoke(self, context, event) -> OperatorReturnItemsSet:
        prefs = self.prefs
        props = self.props
        try:
            restore_dataset_from_json(props, prefs)  # try to restore from local JSON

            # try setting all the properties that are needed for the op
            self._media_files: List[Path] = props.media_files(prefs)
            return self.execute(context)
        except UserFacingError as e:
            self.logger().error(str(e))
            return {"CANCELLED"}

    def contexted_execute(self, context) -> OperatorReturnItemsSet:
        self.operator_thread(
            self,
            _kwargs=ThreadKwargs(
                updated_media_files=self._media_files, dataset_as_dc=self.dataset_as_dc
            ),
        )

        return {"RUNNING_MODAL"}

    def queue_callback(self, context, event, next) -> OptionalOperatorReturnItemsSet:
        is_ok, msg = next
        if msg == "done":
            self.cleanup(context)  # write props (as dataclass) to JSON
            return

        # branch on `is_ok`
        self.logger().info(msg) if is_ok else self.logger().warning(msg)

        # sync props regardless as the updated dataclass is still valid
        self.props.dataset_accessor.from_dataclass(self.dataset_as_dc)

    @staticmethod
    @worker_fn_auto
    def operator_thread(queue, cancel_event, *, _kwargs):
        dataset_as_dc = _kwargs.dataset_as_dc
        status_lookup, accumulator = dataset_as_dc.status_accumulator()

        for file in _kwargs.updated_media_files:
            if cancel_event.is_set():
                return  # simply return as new queue items will not be read anymore
            queue_item = (True, f"Validated media file: '{file}'")

            # check if the status for the file already exists, create new if not
            status = status_lookup.get(str(file), MediaFileStatus(filepath=str(file)))

            # if success is already valid skip new extraction
            if status.needs_reextraction(dataset=dataset_as_dc):
                try:
                    status.extract_from_filepath()  # fill out the dataclass from set filepath
                except UserFacingError as e:
                    queue_item = (False, str(e))  # change queue item and give error msg

            accumulator(status)  # update dataset from new status

            queue.put(queue_item)  # transmit that dataset has been updated

        queue.put((True, "done"))  # signal done
        return
