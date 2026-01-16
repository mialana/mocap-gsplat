import bpy

from pathlib import Path
from typing import ClassVar, List

from ...infrastructure.constants import OperatorIDEnum

from .base_ot import MosplatOperatorBase, OperatorReturnItemsSet, OperatorPollReqs


class Mosplat_OT_prepare_media_directory(MosplatOperatorBase):
    bl_idname = OperatorIDEnum.PREPARE_MEDIA_DIRECTORY
    bl_description = "Prepare media directory for inference."

    poll_reqs = {OperatorPollReqs.PREFS, OperatorPollReqs.PROPS}

    _extensions: ClassVar[List[str]]

    @classmethod
    def poll(cls, context) -> bool:
        if not super().poll(context):
            return False

        prefs = cls.prefs(context)

        extension_set = prefs.media_extension_set
        pref_name = prefs.bl_rna.properties["media_extension_set"].name
        try:
            cls._extensions = [ext.trim() for ext in extension_set.split(",")]
        except IndexError:
            cls.poll_message_set(
                f"Extensions in '{pref_name}' should be separated by commas."
            )
            return False
        return True

    def execute(self, context) -> OperatorReturnItemsSet:
        props = self.props(context)
        prefs = self.prefs(context)

        media_dir = Path(props.current_media_dir)
        extensions = [prefs.media_extension_set.split(".")]

        files = [p for p in media_dir.iterdir() if p.suffix.lower() in extensions]

        if not files:
            self.logger().error("No media files found with the selected extensions.")
            return {"CANCELLED"}

        duration = [self._get_media_duration(p) for p in files]

        if len(set(duration)) != 1:
            self.logger().error("Media files should have the same length.")
            return {"CANCELLED"}

        return {"FINISHED"}

    @classmethod
    def _get_media_duration(cls, filepath: Path) -> int:
        clip = bpy.data.movieclips.load(str(filepath), check_existing=False)
        duration = clip.frame_duration
        cls.logger().info(f"Found a media file of length '{duration}'.")
        bpy.data.movieclips.remove(clip)
        return duration
