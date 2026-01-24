"""native mixin classes. note no import of `bpy`"""

from __future__ import annotations

import os
from typing import (
    ClassVar,
    Type,
    TypeGuard,
    TypeVar,
    Generic,
    Optional,
    TYPE_CHECKING,
    Any,
    TypeAlias,
)
from enum import StrEnum
from dataclasses import fields
import logging
import inspect
import contextlib

from .constants import _MISSING_, DataclassInstance
from .schemas import DeveloperError

S = TypeVar("S")
D = TypeVar("D", bound=DataclassInstance)


class MosplatLogClassMixin:
    """
    this mixin allows classes to have an easily accessible logger.
    this was necessary as creating subclasses of bpy structs would mean that
    all `funcName` logrecords would be the same (e.g. `poll()` or `draw()`, making
    it impossible to distinguish which object class is logging the message,
    and more importantly which is erroring out if it occurs.
    """

    _logger: ClassVar[logging.Logger] = _MISSING_

    @classmethod
    def _create_logger_for_class(cls):
        from ..interfaces import MosplatLoggingInterface

        cls._logger = MosplatLoggingInterface.configure_logger_instance(
            f"{cls.__module__}.logclass{cls.__qualname__}"
        )

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls._create_logger_for_class()  # create logger for subclasses

    @classmethod
    def logger(cls) -> logging.Logger:
        if not cls._logger:
            cls._create_logger_for_class()
        return cls._logger


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
    __id_enum_type__: ClassVar = _MISSING_  # no literals here!

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


class MosplatEncapsulatedContextMixin:
    """
    a mixin that allows for a `context` property within desired instance methods.
    when paired with `contextlib.contextmanager` functions can have a local updated `context` property
    """

    from bpy.types import Context  # local import

    __context: Optional[Context] = None

    @property
    def context(self) -> Context:
        if self.__context is None:  # protect against incorrect usage
            raise DeveloperError("Context has not yet been set yet in this scope.")
        else:
            return self.__context

    @context.setter
    def context(self, context: Optional[Context]):
        self.__context = context

    @contextlib.contextmanager
    def encapsulated_context_block(self, context: Context):
        """ensures `context` is set for code wrapped in a `with` block"""
        self.context = context  # update `context` before block starts
        try:
            yield  # run code in `with`
        finally:  # set to `None` after block finishes
            self.context = None


class MosplatAPAccessorMixin(MosplatLogClassMixin):
    """a mixin class for any class that has access to global preferences"""

    from bpy.types import Context  # local import

    if TYPE_CHECKING:
        from ..core.preferences import Mosplat_AP_Global
    else:
        Mosplat_AP_Global: TypeAlias = Any

    @property
    def prefs(self) -> Mosplat_AP_Global:
        from ..core.checks import check_addonpreferences

        context = getattr(self, "context", None)
        if context is None:
            self.logger().warning("Using fallback context for prefs.", stack_info=True)
            from bpy import context as fallback_context

            context = fallback_context

        return check_addonpreferences(context.preferences)


class MosplatPGAccessorMixin(MosplatLogClassMixin):
    """a mixin class for any class that has access to global properties"""

    from bpy.types import Context  # local import

    if TYPE_CHECKING:
        from ..core.properties import Mosplat_PG_Global
    else:
        Mosplat_PG_Global: TypeAlias = Any

    @property
    def props(self) -> Mosplat_PG_Global:
        from ..core.checks import check_propertygroup

        context = getattr(self, "context", None)
        if context is None:
            self.logger().warning("Using fallback context for props.", stack_info=True)
            from bpy import context as fallback_context

            context = fallback_context

        return check_propertygroup(context.scene)


class MosplatBlPropertyAccessorMixin(
    MosplatEnforceAttributesMixin, MosplatLogClassMixin
):
    """a mixin class for easier access to Blender properties' RNA"""

    bl_rna = _MISSING_

    @classmethod
    def get_prop_name(cls, prop_attrname: str) -> str:
        try:
            return cls.bl_rna.properties[prop_attrname].name
        except KeyError:
            cls.logger().exception(
                f"Tried to retrieve RNA of non-existing property '{prop_attrname}'."
            )
            return f"KEY ERROR. Class: {cls.__qualname__}. Property: {prop_attrname}."  # make error visible and traceable


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
            if hasattr(value, "to_dataclass"):
                value = value.to_dataclass()
            elif isinstance(value, (list, tuple)):
                value = [
                    v.to_dataclass() if hasattr(v, "to_dataclass") else v for v in value
                ]
            kwargs[f.name] = value

        return cls(**kwargs)

    def from_dataclass(self, dc: D) -> None:
        for f in fields(dc):
            value = getattr(dc, f.name)
            target = getattr(self, f.name, None)
            if hasattr(target, "from_dataclass"):
                getattr(target, "from_dataclass")(value)  # nested `PropertyGroup`
            elif hasattr(target, "clear") and hasattr(target, "add"):
                getattr(target, "clear")()  # `CollectionProperty`
                for item_dc in value:
                    item_pg = getattr(target, "add")()
                    if hasattr(item_pg, "from_dataclass"):
                        item_pg.from_dataclass(item_dc)
                    else:
                        item_pg = item_dc
            else:
                setattr(self, f.name, value)
