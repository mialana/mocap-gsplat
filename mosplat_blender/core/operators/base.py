import bpy
from ...infrastructure.mixins import MosplatBlMetaMixin


class MosplatOperatorBase(MosplatBlMetaMixin, bpy.types.Operator):
    prefix_suffix = "OT"
