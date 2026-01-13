import bpy
from ...infrastructure.mixins import MosplatOperatorMixin


class MosplatOperatorBase(MosplatOperatorMixin, bpy.types.Operator):
    pass
