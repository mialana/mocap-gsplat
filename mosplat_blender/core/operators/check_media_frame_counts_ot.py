from pathlib import Path
from typing import ClassVar, Set, Iterable, TYPE_CHECKING, TypeAlias, Any

import threading
from queue import Queue

from .base_ot import MosplatOperatorBase, OperatorReturnItemsSet, OperatorPollReqs
from ..checks import check_data_output_dir

from ...infrastructure.constants import OperatorIDEnum
from ...interfaces.media_io_interface import (
    MosplatMediaIOInterface,
    MediaProcessStatus,
)

if TYPE_CHECKING:
    from ..properties import Mosplat_PG_MediaItem
else:
    Mosplat_PG_MediaItem: TypeAlias = Any


class Mosplat_OT_check_media_frame_counts(MosplatOperatorBase):
    bl_idname = OperatorIDEnum.CHECK_MEDIA_FRAME_COUNTS
    bl_description = (
        "Check frame counts of all media files found in given media directory."
    )

    poll_reqs = {OperatorPollReqs.PREFS, OperatorPollReqs.PROPS}

    _extensions: ClassVar[Set[str]]

    _media_dir_path: ClassVar[Path]
    _data_output_dir: ClassVar[Path]

    _files: ClassVar[Iterable[Path]]

    @classmethod
    def poll(cls, context) -> bool:
        if not super().poll(context):
            return False

        prefs = cls.prefs(context)
        props = cls.props(context)

        extension_set_str: str = prefs.media_extension_set
        try:
            cls._extensions = set(
                [ext.strip().lower() for ext in extension_set_str.split(",")]
            )
        except IndexError:
            pref_name = prefs.get_prop_name("media_extension_set")
            cls.poll_message_set(
                f"Extensions in '{pref_name}' should be separated by commas."
            )
            return False

        cls._media_dir_path = Path(props.current_media_dir)

        # validate the data output dir preference
        try:
            cls._data_output_dir = check_data_output_dir(context)
        except AttributeError as e:
            cls.poll_message_set(str(e))
            return False

        cls._files = [
            p
            for p in cls._media_dir_path.iterdir()
            if p.suffix.lower() in cls._extensions
        ]

        if not cls._files:
            cls.poll_message_set(
                f"No files found in the media directory with the preferred extensions: `{extension_set_str}`\n"
                f"Configure '{prefs.get_prop_name('media_extension_set')}' if needed."
            )
            return False
        return True

    def execute(self, context) -> OperatorReturnItemsSet:
        if not MosplatMediaIOInterface.initialized:
            MosplatMediaIOInterface.initialize(
                self._media_dir_path, self._data_output_dir
            )

        self.props(context).found_media_files.clear()

        self._queue = Queue()

        self._thread = threading.Thread(target=self._process_files_thread, daemon=True)
        self._thread.start()

        self._timer = self.wm(context).event_timer_add(
            time_step=0.1, window=context.window
        )
        self.wm(context).modal_handler_add(self)  # start timer polling here

        return {"RUNNING_MODAL"}

    def _process_files_thread(self):
        self.logger().info(
            f"Reading frame counts of files within '{self._media_dir_path}'. This might take a while..."
        )
        for fp in self._files:
            for status in MosplatMediaIOInterface.process_media_file(fp):
                self._queue.put(("status", status))
                if not status.ok:
                    self._queue.put(("done", False))
                    return  # stop processing
        self._queue.put(("done", True))

    def modal(self, context, event) -> OperatorReturnItemsSet:
        if event.type in {"RIGHTMOUSE", "ESC"}:
            self._cleanup(context)
            return {"CANCELLED"}
        elif event.type != "TIMER":
            return {"PASS_THROUGH"}

        props = self.props(context)

        while not self._queue.empty():
            tag, payload = self._queue.get_nowait()

            if tag == "status":
                status: MediaProcessStatus = payload

                media: Mosplat_PG_MediaItem = props.found_media_files.add()
                media.filepath = str(status.filepath)
                media.frame_count = status.frame_count

                if status.ok:
                    self.logger().info(status.message)
                else:
                    self.logger().error(status.message)

                if context.area:
                    context.area.tag_redraw()

            elif tag == "done":
                self._cleanup(context)
                MosplatMediaIOInterface.update_metadata_json()
                if payload:
                    props.do_media_durations_all_match = True
                    props.collective_media_frame_count = (
                        MosplatMediaIOInterface.metadata.collective_frame_count
                    )
                    self.logger().info(
                        f"Frame count of media files in '{self._media_dir_path}' all match."
                    )
                    return {"FINISHED"}
                else:
                    props.do_media_durations_all_match = False
                    props.collective_media_frame_count = -1
                    self.logger().warning(
                        f"'{self._media_dir_path}' contains media files of different frame counts."
                    )
                    return {"CANCELLED"}

        return {"RUNNING_MODAL", "PASS_THROUGH"}
