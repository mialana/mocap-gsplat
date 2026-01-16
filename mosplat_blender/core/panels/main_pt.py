from ...infrastructure.constants import PanelIDEnum

from .base import MosplatPanelBase


class Mosplat_PT_Main(MosplatPanelBase):
    bl_idname = PanelIDEnum.MAIN
    bl_description = "Main panel holding all Mosplat panels"
