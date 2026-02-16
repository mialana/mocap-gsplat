from __future__ import annotations

import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

from ..infrastructure.schemas import MediaFileStatus, MediaIOMetadata
from .base_ot import MosplatOperatorBase


class ProcessKwargs(NamedTuple):
    updated_media_files: List[Path]
    data: MediaIOMetadata


class Mosplat_OT_validate_media_statuses(
    MosplatOperatorBase[Tuple[str, str, Optional[MediaIOMetadata]], ProcessKwargs]
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
            pwargs=ProcessKwargs(updated_media_files=self._media_files, data=self.data),
        )

        return "RUNNING_MODAL"

    def _queue_callback(self, pkg, event, next):
        status, msg, new_data = next
        props = pkg.props

        # sync props regardless as the updated dataclass is still valid
        if new_data:
            self.data = new_data
            self.sync_to_props(props)

        if status == "done":
            first, _ = props.frame_range_
            props.frame_range[0] = first  # trigger frame range update

        return super()._queue_callback(pkg, event, next)

    @staticmethod
    def _operator_subprocess(queue, cancel_event, *, pwargs):
        import torch
        from torchcodec.decoders import VideoDecoder, VideoStreamMetadata

        files, data = pwargs

        torchcodec_device: str = (
            "cuda"
            if not sys.platform == "win32" and torch.cuda.is_available()
            else "cpu"  # torchcodec does not ship cuda-enabled wheels through PyPI on Windows
        )

        status_lookup, accumulator = data.status_accumulator()

        for file in files:
            if cancel_event.is_set():
                return  # simply return as new queue items will not be read anymore
            queue_item = ("update", f"Validated media file: '{file}'", data)

            # check if the status for the file already exists, create new if not
            status = status_lookup.get(str(file), MediaFileStatus(filepath=str(file)))

            # if success is already valid skip new extraction
            if status.needs_reextraction(data=data):
                try:
                    # create decoder here so torchcodec only needs to be imported once
                    decoder: VideoDecoder = VideoDecoder(
                        status.filepath, device=torchcodec_device
                    )
                    metadata: VideoStreamMetadata = decoder.metadata
                    status.from_torchcodec(metadata)

                    del decoder  # clean up resources
                except RuntimeError as e:
                    status.mark_invalid()
                    # change queue item and give error msg
                    queue_item = ("warning", str(e), data)

            accumulator(status)  # update metadata from new status
            queue.put(queue_item)  # transmit that metadata has been updated

        # get final validity
        result = ("success" if data.is_valid_media_directory else "failure").upper()
        queue.put(("done", f"Validation complete: {result}", None))  # signal done
        return


def process_entrypoint(*args, **kwargs):
    Mosplat_OT_validate_media_statuses._operator_subprocess(*args, **kwargs)
