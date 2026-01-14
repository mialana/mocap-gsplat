from ...infrastructure.constants import OperatorIDEnum, PanelIDEnum

from .base import MosplatPanelBase


class Mosplat_PT_Main(MosplatPanelBase):
    bl_idname = PanelIDEnum.MAIN

    @classmethod
    def poll(cls, context):
        return True
