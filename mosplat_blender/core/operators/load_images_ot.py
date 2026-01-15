from bpy.props import StringProperty

from ...infrastructure.constants import OperatorIDEnum

from .base import MosplatOperatorBase, OperatorReturnItemsSet


class Mosplat_OT_load_images(MosplatOperatorBase):
    """Load images from a directory"""

    bl_idname = OperatorIDEnum.LOAD_IMAGES

    dirpath: StringProperty(
        subtype="DIR_PATH"
    )  # pyright: ignore[reportInvalidTypeForm]

    def execute(self, context) -> OperatorReturnItemsSet:
        if not (props := self.props(context)):
            return {"CANCELLED"}

        return {"FINISHED"}

    def invoke(self, context, event) -> OperatorReturnItemsSet:
        if not (props := self.props(context)):
            return {"CANCELLED"}

        wm = context.window_manager
        if not wm:
            return {"CANCELLED"}

        wm.fileselect_add(self)
        return {"RUNNING_MODAL"}
