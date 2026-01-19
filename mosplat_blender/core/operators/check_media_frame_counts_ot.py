from pathlib import Path
from typing import ClassVar, Set, List


from .base_ot import MosplatOperatorBase, OperatorReturnItemsSet, OperatorPollReqs
from ..checks import check_data_output_dir

from ...infrastructure.constants import OperatorIDEnum
from ...interfaces import MosplatMediaIOInterface


class Mosplat_OT_check_media_frame_counts(MosplatOperatorBase):
    bl_idname = OperatorIDEnum.CHECK_MEDIA_FRAME_COUNTS
    bl_description = (
        "Check frame counts of all media files found in given media directory."
    )

    poll_reqs = {OperatorPollReqs.PREFS, OperatorPollReqs.PROPS}

    _extensions: ClassVar[Set[str]]

    _media_dir_path: ClassVar[Path]
    _data_output_dir: ClassVar[Path]

    _files: ClassVar[List[Path]]

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
            cls._media_dir_path = check_data_output_dir(context)
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
        props = self.props(context)
        prefs = self.prefs(context)

        def _on_failure(msg: str) -> OperatorReturnItemsSet:
            props.do_media_durations_all_match = False
            props.computed_media_frame_count = -1
            self.logger().error(msg)
            return {"CANCELLED"}

        props.found_media_files.clear()

        self.logger().info(
            f"Reading frame counts of files within '{self._media_dir_path}'. This might take a while..."
        )

        if not MosplatMediaIOInterface.initialized:
            MosplatMediaIOInterface.initialize(
                self._media_dir_path, self._data_output_dir
            )

        frame_counts = []

        # media = found_media_files.add()
        # media.filepath = str(filepath)
        # media.frame_count = frame_count
        #             cls.logger().debug(
        #     f"Read video file '{filepath}' with the duration '{frame_count}' frames ({method})."
        # )

        if len(set(frame_counts)) != 1:
            return _on_failure(
                f"Media files within '{self._media_dir_path}' should all have the same frame count."
            )

        props.do_media_durations_all_match = True
        props.computed_media_frame_count = frame_counts[0]
        self.logger().info(f"'{self._media_dir_path}' is a valid media directory.")

        return {"FINISHED"}
