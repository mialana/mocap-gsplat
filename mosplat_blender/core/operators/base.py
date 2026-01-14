import bpy
from bpy.types import Context

from typing import Union

from ..checks import check_props_safe, check_prefs_safe
from ..properties import Mosplat_PG_Global
from ..preferences import Mosplat_AP_Global
from ...infrastructure.mixins import MosplatOperatorMixin


class MosplatOperatorBase(MosplatOperatorMixin, bpy.types.Operator):
    def props(self, context: Context) -> Union[Mosplat_PG_Global, None]:
        return check_props_safe(context)

    def prefs(self, context: Context) -> Union[Mosplat_AP_Global, None]:
        return check_prefs_safe(context)
