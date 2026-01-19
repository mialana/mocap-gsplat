from pathlib import Path
from typing import ClassVar, List

from ...infrastructure.constants import OperatorIDEnum

from .base_ot import MosplatOperatorBase, OperatorReturnItemsSet, OperatorPollReqs


class Mosplat_OT_check_media_frame_counts(MosplatOperatorBase):
    bl_idname = OperatorIDEnum.CHECK_MEDIA_FRAME_COUNTS
    bl_description = (
        "Check frame counts of all media files found in given media directory."
    )

    poll_reqs = {OperatorPollReqs.PREFS, OperatorPollReqs.PROPS}

    _extensions: ClassVar[List[str]]

    @classmethod
    def poll(cls, context) -> bool:
        if not super().poll(context):
            return False

        prefs = cls.prefs(context)

        extension_set_str: str = prefs.media_extension_set
        pref_name = prefs.bl_rna.properties["media_extension_set"].name
        try:
            cls._extensions = [
                ext.strip().lower() for ext in extension_set_str.split(",")
            ]
        except IndexError:
            cls.poll_message_set(
                f"Extensions in '{pref_name}' should be separated by commas."
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

        media_dir_path = Path(props.current_media_dir)
        media_extension_set_str: str = prefs.media_extension_set

        extensions = [ext.strip().lower() for ext in media_extension_set_str.split(",")]

        files = [p for p in media_dir_path.iterdir() if p.suffix.lower() in extensions]

        if not files:
            return _on_failure(
                f"No files found in the media directory with the preferred extensions: `{media_extension_set_str}`\n"
                f"Configure '{prefs.bl_rna.properties['media_extension_set'].name}' if needed."
            )

        self.logger().info(
            f"Reading frame counts of files within '{media_dir_path}'. This might take a while..."
        )

        frame_counts = [
            self._get_media_duration(p, props.found_media_files) for p in files
        ]

        if len(set(frame_counts)) != 1:
            return _on_failure(
                f"Media files within '{media_dir_path}' should all have the same frame count."
            )

        props.do_media_durations_all_match = True
        props.computed_media_frame_count = frame_counts[0]
        self.logger().info(f"'{media_dir_path}' is a valid media directory.")

        return {"FINISHED"}

    @classmethod
    def _get_media_duration(cls, filepath: Path, found_media_files) -> int:
        import cv2

        def _cleanup(method: str):
            cap.release()

            cls.logger().debug(
                f"Read video file '{filepath}' with the duration '{frame_count}' frames ({method})."
            )

            media = found_media_files.add()
            media.filepath = str(filepath)
            media.frame_count = frame_count

            return frame_count

        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open media file: {filepath}")

        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)  # seek to end
        duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps > 0 and duration_ms > 0:
            frame_count = int(round((duration_ms / 1000.0) * fps))
            if frame_count > 0:
                return _cleanup("fps + duration metadata")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if 0 < frame_count < 2**32 - 1:
            return _cleanup("frame count metadata")

        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.0)  # return seek to start

        frame_count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            frame_count += 1

        return _cleanup("manual")
