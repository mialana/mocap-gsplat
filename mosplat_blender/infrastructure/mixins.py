"""native mixin classes. note no import of `bpy`"""

from __future__ import annotations

import os
from typing import ClassVar, Type, TypeGuard, TypeVar, TypeAlias
from enum import StrEnum
import logging
import inspect

from ..interfaces.logging_interface import MosplatLoggingInterface

from .constants import _MISSING_

S = TypeVar("S")


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
    def create_logger_for_class(cls):
        cls._logger = MosplatLoggingInterface.configure_logger_instance(
            f"{cls.__module__}.logclass{cls.__qualname__}"
        )

    @classmethod
    def logger(cls) -> logging.Logger:
        if not cls._logger:
            cls.create_logger_for_class()
        return cls._logger


class MosplatBlTypeMixin(MosplatLogClassMixin):
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
    id_enum_type: ClassVar = _MISSING_  # no literals here!

    @classmethod
    def at_registration(cls):
        """
        ran at registration time of the class.
        useful for differentiating 'abstract' base class behavior from registered classes
        """
        cls.create_logger_for_class()

        if os.getenv("MOSPLAT_TESTING"):
            cls._enforce_mixin_attributes()
            cls.guard_type_of_bl_idname(cls.bl_idname, cls.id_enum_type)

    @classmethod
    def _enforce_mixin_attributes(cls):
        """enforce all required attributes are defined"""

        # `getmembers_static` with custom predicate
        if missing := inspect.getmembers_static(cls, lambda val: val is _MISSING_):
            for attr_name, _ in missing:
                raise AttributeError(
                    f"`{cls.__name__}` does not define required mixin variable: `{attr_name}`"
                )

    @staticmethod
    def guard_type_of_bl_idname(bl_idname, id_enum_type: Type[S]) -> TypeGuard[S]:
        """guards `bl_idname` with its narrowed type"""
        if not issubclass(id_enum_type, StrEnum):
            raise TypeError(
                f"`{__name__}` defines `id_enum_type` with incorrect type.\nExpected: {Type[StrEnum]}, \nActual: {bl_idname}"
            )
        if not isinstance(bl_idname, id_enum_type):
            raise TypeError(
                f"`{__name__}` defines `bl_idname` with incorrect type.\nExpected: {Type[id_enum_type]}, \nActual: {bl_idname}"
            )
        return True
