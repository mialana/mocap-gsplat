from bpy.types import Panel, UILayout, Context

from typing import ClassVar, Set, Union
from enum import Enum
from functools import partial

from ..checks import (
    check_addonpreferences,
    check_propertygroup,
    check_prefs_safe,
    check_props_safe,
)
from ..properties import Mosplat_PG_Global
from ..preferences import Mosplat_AP_Global

from ...infrastructure.mixins import (
    MosplatBlTypeMixin,
    MosplatPGAccessorMixin,
    MosplatAPAccessorMixin,
)
from ...infrastructure.constants import PanelIDEnum


class PanelPollReqs(Enum):
    """Custom enum in case operator does not require use of one poll requirement"""

    PREFS = partial(lambda cls, context: check_prefs_safe(context))
    PROPS = partial(lambda cls, context: check_props_safe(context))


class MosplatPanelBase(
    MosplatBlTypeMixin, MosplatPGAccessorMixin, MosplatAPAccessorMixin, Panel
):
    __id_enum_type__ = PanelIDEnum

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = PanelIDEnum._category()

    __poll_reqs__: ClassVar[Union[Set[PanelPollReqs], None]] = {
        PanelPollReqs.PREFS,
        PanelPollReqs.PROPS,
    }

    @classmethod
    def at_registration(cls):
        super().at_registration()
        if cls.guard_type_of_bl_idname(cls.bl_idname, cls.__id_enum_type__):
            cls.bl_label = PanelIDEnum.label_factory(cls.bl_idname)

    @classmethod
    def poll(cls, context) -> bool:
        return (
            all(req.value(cls, context) for req in cls.__poll_reqs__)
            if cls.__poll_reqs__
            else True
        )

    def draw(self, context: Context):
        if not (layout := self.layout):
            return

        return self.draw_with_layout(context, layout)

    def draw_with_layout(self, context: Context, layout: UILayout):
        """layout will always exist with this function"""
        ...
