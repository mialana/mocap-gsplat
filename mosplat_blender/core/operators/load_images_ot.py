from bpy.props import StringProperty

from ...infrastructure.constants import OperatorIDEnum

from .base_ot import MosplatOperatorBase, OperatorReturnItemsSet, OperatorPollReqs


class Mosplat_OT_load_images(MosplatOperatorBase):
    bl_idname = OperatorIDEnum.LOAD_IMAGES
    bl_description = "Load input images"

    poll_reqs = {OperatorPollReqs.PROPS, OperatorPollReqs.WINDOW_MANAGER}

    directory: StringProperty(
        subtype="DIR_PATH"
    )  # pyright: ignore[reportInvalidTypeForm]

    def execute(self, context) -> OperatorReturnItemsSet:
        self.logger().debug(f"Selected directory path at '{self.directory}'")

        # re-sync to global properties
        self.props(context).current_image_dir = self.directory

        return {"FINISHED"}

    def invoke(self, context, event) -> OperatorReturnItemsSet:
        # sync member variable with corresponding variable on global properties
        self.directory = self.props(context).current_image_dir

        # self.wm(context).invoke_props_dialog(self, width=500)
        self.wm(context).fileselect_add(self)
        return {"RUNNING_MODAL"}
