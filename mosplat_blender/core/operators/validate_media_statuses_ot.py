from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple, Tuple

from core.operators.base_ot import MosplatOperatorBase
from infrastructure.schemas import MediaFileStatus, MediaIODataset


class ThreadKwargs(NamedTuple):
    updated_media_files: List[Path]
    dataset_as_dc: MediaIODataset


class Mosplat_OT_validate_media_statuses(
    MosplatOperatorBase[Tuple[bool, str], ThreadKwargs]
):
    def _contexted_invoke(self, pkg, event):
        prefs = pkg.prefs
        props = pkg.props

        # try setting all the properties that are needed for the op
        self._media_files: List[Path] = props.media_files(prefs)
        return self.execute_with_package(pkg)

    def _contexted_execute(self, pkg):
        self.launch_thread(
            pkg.context,
            twargs=ThreadKwargs(
                updated_media_files=self._media_files, dataset_as_dc=self.data
            ),
        )

        return "RUNNING_MODAL"

    def _queue_callback(self, pkg, event, next):
        is_ok, msg = next
        if msg == "done":
            return "FINISHED"

        # branch on `is_ok`
        self.logger.info(msg) if is_ok else self.logger.warning(msg)

        # sync props from the dataclass that was updated within the thread
        self._sync_to_props(pkg.props)
        return "RUNNING_MODAL"

    @staticmethod
    def _operator_thread(queue, cancel_event, *, twargs):
        dataset_as_dc = twargs.dataset_as_dc
        status_lookup, accumulator = dataset_as_dc.status_accumulator()

        for file in twargs.updated_media_files:
            if cancel_event.is_set():
                return  # simply return as new queue items will not be read anymore
            queue_item = (True, f"Validated media file: '{file}'")

            # check if the status for the file already exists, create new if not
            status = status_lookup.get(str(file), MediaFileStatus(filepath=str(file)))

            # if success is already valid skip new extraction
            if status.needs_reextraction(dataset=dataset_as_dc):
                try:
                    status.extract_from_filepath()  # fill out the dataclass from set filepath
                except OSError as e:
                    queue_item = (False, str(e))  # change queue item and give error msg

            accumulator(status)  # update dataset from new status

            queue.put(queue_item)  # transmit that dataset has been updated

        queue.put((True, "done"))  # signal done
        return
