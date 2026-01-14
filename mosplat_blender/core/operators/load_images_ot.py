from ...infrastructure.constants import OperatorIDEnum

from .base import MosplatOperatorBase, OperatorReturnItemsSet


class Mosplat_OT_load_images(MosplatOperatorBase):
    """Load images from a directory"""

    bl_idname = OperatorIDEnum.LOAD_IMAGES

    _thread = None

    def execute(self, context) -> OperatorReturnItemsSet:
        if not (props := self.props(context)):
            return {"CANCELLED"}

        return {"FINISHED"}
