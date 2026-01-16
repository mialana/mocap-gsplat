from bpy.props import StringProperty

from ...infrastructure.constants import OperatorIDEnum

from .base_ot import MosplatOperatorBase, OperatorReturnItemsSet, OperatorPollReqs


class Mosplat_OT_select_media_directory(MosplatOperatorBase):
    bl_idname = OperatorIDEnum.SELECT_MEDIA_DIRECTORY
    bl_description = "Select a directory that contains video files (*.avi, *.mp4, or *.mov) to be converted to gaussian splat data"

    poll_reqs = {OperatorPollReqs.PROPS, OperatorPollReqs.WINDOW_MANAGER}

    directory: StringProperty(
        subtype="DIR_PATH"
    )  # pyright: ignore[reportInvalidTypeForm]

    def execute(self, context) -> OperatorReturnItemsSet:
        self.logger().debug(f"Selected directory path at '{self.directory}'")

        self._validate_directory_selection()

        # re-sync to global properties
        self.props(context).current_media_dir = self.directory

        return {"FINISHED"}

    def invoke(self, context, event) -> OperatorReturnItemsSet:
        # sync member variable with corresponding variable on global properties
        self.directory = self.props(context).current_media_dir

        self.wm(context).fileselect_add(self)
        return {"RUNNING_MODAL"}

    def _validate_directory_selection(self):
        pass
