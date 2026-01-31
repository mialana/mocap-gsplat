"""native mixin classes. note no import of `bpy`"""

from __future__ import annotations

import os
from typing import (
    ClassVar,
    Type,
    TypeGuard,
    TypeVar,
    Generic,
    TYPE_CHECKING,
    List,
    NamedTuple,
    Optional,
)
from enum import StrEnum
from dataclasses import fields
import logging
import inspect


from .protocols import SupportsCollectionProperty, SupportsDataclass
from .constants import _MISSING_
from .schemas import DeveloperError

S = TypeVar("S")

D = TypeVar("D", bound=SupportsDataclass)  # dataclass equivalent of the property group
ChildD = TypeVar("ChildD", bound=SupportsDataclass)  # dataclass that is a property

if TYPE_CHECKING:
    from bpy.types import Context
    from ..core.preferences import Mosplat_AP_Global
    from ..core.properties import Mosplat_PG_Global


class MosplatLogClassMixin:
    """
    this mixin allows classes to have an easily accessible logger.
    this was necessary as creating subclasses of bpy structs would mean that
    all `funcName` logrecords would be the same (e.g. `poll()` or `draw()`, making
    it impossible to distinguish which object class is logging the message,
    and more importantly which is erroring out if it occurs.
    """

    class_logger: ClassVar[logging.Logger] = _MISSING_

    @classmethod
    def _create_logger_for_class(cls, label: Optional[str] = None):
        from ..interfaces import MosplatLoggingInterface

        cls.class_logger = MosplatLoggingInterface.configure_logger_instance(
            f"{cls.__module__}.logclass{label if label else cls.__qualname__}"
        )

    @classmethod
    def __init_subclass__(cls, label: Optional[str] = None, /, **kwargs):
        super().__init_subclass__(**kwargs)

        cls._create_logger_for_class(label)  # create logger for subclasses

    @property
    def logger(self) -> logging.Logger:
        cls = self.__class__
        if cls.class_logger is _MISSING_:
            cls._create_logger_for_class()
        return cls.class_logger


class MosplatEnforceAttributesMixin:

    @classmethod
    def at_registration(cls):
        """
        ran at registration time of the class.
        useful for differentiating 'abstract' base class behavior from registered classes
        """
        if "MOSPLAT_TESTING" in os.environ:
            cls._enforce_mixin_attributes()

    @classmethod
    def _enforce_mixin_attributes(cls):
        """enforce all required attributes are defined"""

        # `getmembers_static` with custom predicate
        if missing := inspect.getmembers_static(cls, lambda val: val is _MISSING_):
            for attr_name, _ in missing:
                raise DeveloperError(
                    f"`{cls.__name__}` does not define required mixin variable: `{attr_name}`"
                )


class MosplatBlTypeMixin(MosplatLogClassMixin, MosplatEnforceAttributesMixin):
    """
    a mixin base class that allows for standardization of common metadata between
    multiple subclasses of a Blender type.

    namely, panel classes have strict requirements for `bl_idname`,
    and operators have requirements for `bl_label`.
    but the definition is repetitive, hence the usefulness of a mixin.

    specifically, with `at_registration`, the proper conventions for Blender types
    can be enforced when registering the class, and does not have to be hardcoded
    on each individual class instance.

    furthermore, these enforcements ensure that in development,
    the defined variables are of the type and identity that they should be,
    in the effort of increasing consistency in the codebase.
    """

    bl_idname: str = _MISSING_
    bl_description: str = _MISSING_
    __id_enum_type__: ClassVar[type] = _MISSING_  # no literals here!

    @classmethod
    def at_registration(cls):
        super().at_registration()

        if "MOSPLAT_TESTING" in os.environ:
            cls.guard_type_of_bl_idname(cls.bl_idname, cls.__id_enum_type__)

    @staticmethod
    def guard_type_of_bl_idname(bl_idname, __id_enum_type__: Type[S]) -> TypeGuard[S]:
        """guards `bl_idname` with its narrowed type"""
        if not issubclass(__id_enum_type__, StrEnum):
            raise DeveloperError(
                f"`{__name__}` defines `__id_enum_type__` with incorrect type.\nExpected: {Type[StrEnum]}, \nActual: {bl_idname}"
            )
        if not isinstance(bl_idname, __id_enum_type__):
            raise DeveloperError(
                f"`{__name__}` defines `bl_idname` with incorrect type.\nExpected: {Type[__id_enum_type__]}, \nActual: {bl_idname}"
            )
        return True


class MosplatAPAccessorMixin(MosplatLogClassMixin):
    """a mixin class for any class that has access to global preferences"""

    from bpy.types import Context  # local import

    if TYPE_CHECKING:
        from ..core.preferences import Mosplat_AP_Global

    @staticmethod
    def prefs(context) -> Mosplat_AP_Global:
        from ..core.checks import check_addonpreferences

        return check_addonpreferences(context.preferences)


class MosplatPGAccessorMixin(MosplatLogClassMixin):
    """a mixin class for any class that has access to global properties"""

    from bpy.types import Context  # local import

    if TYPE_CHECKING:
        from ..core.properties import Mosplat_PG_Global

    @staticmethod
    def props(context: Context) -> Mosplat_PG_Global:
        from ..core.checks import check_propertygroup

        return check_propertygroup(context.scene)


class MosplatBlPropertyAccessorMixin(
    MosplatEnforceAttributesMixin, MosplatLogClassMixin
):
    """a mixin class for easier access to Blender properties' RNA"""

    from bpy.types import BlenderRNA

    bl_rna: ClassVar[BlenderRNA]

    @classmethod
    def get_prop_name(cls, prop_attrname: str) -> str:
        try:
            return cls.bl_rna.properties[prop_attrname].name
        except KeyError:
            cls.class_logger.error(
                f"Tried to retrieve RNA name of non-existing prop: '{prop_attrname}'."
            )
            return f"KEY ERROR. Class: {cls.__qualname__}. Property: {prop_attrname}."  # make error visible and traceable

    @classmethod
    def unset_prop(cls, prop_attrname: str):
        try:
            cls.bl_rna.property_unset(prop_attrname)
        except KeyError:
            cls.class_logger.error(
                f"Tried to retrieve RNA of non-existing property '{prop_attrname}'."
            )


class CtxPackage(NamedTuple):
    context: Context
    prefs: Mosplat_AP_Global
    props: Mosplat_PG_Global


class MosplatContextAccessorMixin(
    MosplatBlTypeMixin, MosplatAPAccessorMixin, MosplatPGAccessorMixin
):

    if TYPE_CHECKING:
        from bpy.types import Context

    @classmethod
    def package(cls, context: Context) -> CtxPackage:
        """convenience method to package context"""
        return CtxPackage(
            context=context, prefs=cls.prefs(context), props=cls.props(context)
        )


class MosplatDataclassInteropMixin(Generic[D], MosplatEnforceAttributesMixin):
    """
    a mixin that automates the transformation to/from a dataclass.
    used for Blender `PropertyGroup` classes."""

    __dataclass_type__: Type[D] = _MISSING_

    def to_dataclass(self) -> D:
        cls = self.__dataclass_type__

        kwargs = {}
        for f in fields(cls):
            value = getattr(self, f.name)
            if isinstance(value, MosplatDataclassInteropMixin):
                value = value.to_dataclass()
            elif isinstance(value, SupportsCollectionProperty):
                value = self.collection_property_to_dataclass_list(value)
            kwargs[f.name] = value

        return cls(**kwargs)

    def from_dataclass(self, dc: D) -> None:
        for f in fields(dc):
            value = getattr(dc, f.name)
            target = getattr(self, f.name, None)
            if isinstance(target, MosplatDataclassInteropMixin):
                target.from_dataclass(value)
            elif isinstance(target, SupportsCollectionProperty):
                target.clear()  # clear out the old collection property
                for item_dc in value:
                    item_pg = target.add()
                    if isinstance(item_pg, MosplatDataclassInteropMixin):
                        item_pg.from_dataclass(item_dc)
                    else:
                        item_pg = item_dc
            else:
                setattr(self, f.name, value)

    @staticmethod
    def collection_property_to_dataclass_list(
        cp: SupportsCollectionProperty[MosplatDataclassInteropMixin[ChildD]],
    ) -> List[ChildD]:
        return [v.to_dataclass() for v in cp]
