import bpy
from bpy.types import Context

from .preferences import Mosplat_AP_Global
from .properties import Mosplat_PG_Global
from .checks import check_addonpreferences

from ..infrastructure.mixins import MosplatPanelMixin, MosplatOperatorMixin


class MosplatBlTypeBase:
    def props(self, context: Context) -> Mosplat_PG_Global | None:
        pass

    def prefs(self, context: Context) -> Mosplat_AP_Global | None:
        try:
            return check_addonpreferences(context.preferences)
        except KeyError:
            return None
        except RuntimeError:
            # most member functions that bpy types define should not have `None` for `Context`
            # so this connotates developer logic error and should be raised
            raise AssertionError("We are doing something wrong!")


class MosplatOperatorBase(MosplatBlTypeBase, MosplatOperatorMixin, bpy.types.Operator):
    pass


class MosplatPanelBase(MosplatBlTypeBase, MosplatPanelMixin, bpy.types.Panel):
    pass
