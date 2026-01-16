from bpy.props import StringProperty

from ...infrastructure.constants import OperatorIDEnum

from .base import MosplatOperatorBase, OperatorReturnItemsSet, OperatorPollReqs


class Mosplat_OT_load_images(MosplatOperatorBase):
    """Load images from a directory"""

    bl_idname = OperatorIDEnum.LOAD_IMAGES
    bl_description = "Load input images"

    poll_reqs = {OperatorPollReqs.PROPS, OperatorPollReqs.WINDOW_MANAGER}

    dirpath: StringProperty(
        subtype="DIR_PATH"
    )  # pyright: ignore[reportInvalidTypeForm]

    def execute(self, context) -> OperatorReturnItemsSet:
        return {"FINISHED"}

    def invoke(self, context, event) -> OperatorReturnItemsSet:
        self.wm(context).fileselect_add(self)
        return {"RUNNING_MODAL"}
