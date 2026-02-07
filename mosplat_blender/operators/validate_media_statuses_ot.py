from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

from infrastructure.macros import try_access_path
from infrastructure.schemas import MediaFileStatus, MediaIODataset
from operators.base_ot import MosplatOperatorBase


class ProcessKwargs(NamedTuple):
    updated_media_files: List[Path]
    dataset_as_dc: MediaIODataset


class Mosplat_OT_validate_media_statuses(
    MosplatOperatorBase[Tuple[bool, str, Optional[MediaIODataset]], ProcessKwargs]
):
    def _contexted_invoke(self, pkg, event):
        prefs = pkg.prefs
        props = pkg.props

        # try setting all the properties that are needed for the op
        self._media_files: List[Path] = props.media_files(prefs)
        return self.execute_with_package(pkg)

    def _contexted_execute(self, pkg):
        self.launch_subprocess(
            pkg.context,
            pwargs=ProcessKwargs(
                updated_media_files=self._media_files, dataset_as_dc=self.data
            ),
        )

        return "RUNNING_MODAL"

    def _queue_callback(self, pkg, event, next):
        is_ok, msg, new_data = next
        if msg == "done":
            return "FINISHED"

        # branch on `is_ok`
        self.logger.info(msg) if is_ok else self.logger.warning(msg)

        # sync props from the dataclass that was updated within the thread
        if new_data:
            self.data = new_data
            self._sync_to_props(pkg.props)
        return "RUNNING_MODAL"

    @staticmethod
    def _operator_subprocess(queue, cancel_event, *, pwargs):
        from torch import cuda, device
        from torchcodec.decoders import VideoDecoder, VideoStreamMetadata

        dev = device("cuda" if cuda.is_available() else "cpu")

        dataset_as_dc = pwargs.dataset_as_dc
        status_lookup, accumulator = dataset_as_dc.status_accumulator()

        for file in pwargs.updated_media_files:
            if cancel_event.is_set():
                return  # simply return as new queue items will not be read anymore
            queue_item = (True, f"Validated media file: '{file}'", dataset_as_dc)

            # check if the status for the file already exists, create new if not
            status = status_lookup.get(str(file), MediaFileStatus(filepath=str(file)))

            # if success is already valid skip new extraction
            if status.needs_reextraction(dataset=dataset_as_dc):
                try:
                    decoder: VideoDecoder = VideoDecoder(status.filepath, device=dev)
                    metadata: VideoStreamMetadata = decoder.metadata
                    status.from_torchcodec(metadata)
                except RuntimeError as e:
                    status.mark_invalid()
                    # change queue item and give error msg
                    queue_item = (False, str(e), dataset_as_dc)

            accumulator(status)  # update dataset from new status

            queue.put(queue_item)  # transmit that dataset has been updated

        queue.put((True, "done", None))  # signal done
        return


def process_entrypoint(*args, **kwargs):
    Mosplat_OT_validate_media_statuses._operator_subprocess(*args, **kwargs)
