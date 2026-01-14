"""native mixin classes. note no import of `bpy`"""

from __future__ import annotations

from typing import ClassVar, Type
from enum import StrEnum
import logging
import inspect

from ..interfaces.logging_interface import MosplatLoggingInterface

from .constants import _MISSING_


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
        cls._logger = MosplatLoggingInterface.configure_logger_instance(
            f"{cls.__module__}.logclass{cls.__qualname__}"
        )

    @classmethod
    def logger(cls) -> logging.Logger:
        if not cls._logger:
            cls._create_logger_for_class()
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
    """

    bl_idname: str = _MISSING_
    id_enum_type: StrEnum = _MISSING_  # enforces enum usage so no literals

    @classmethod
    def at_registration(cls):
        """
        ran at registration time of the class.
        useful for differentiating 'abstract' base class behavior from registered classes
        """
        cls._create_logger_for_class()
        cls._enforce_mixin_attributes()

        if not isinstance(cls.bl_idname, Type(cls.id_enum_type)):
            raise TypeError(
                f"`{cls.__name__}` defines id_name with incorrect type.\nExpected: {Type[cls.id_enum_type]}, \nActual: {Type(cls.bl_idname)}"
            )

    @classmethod
    def _enforce_mixin_attributes(cls):
        """enforce all required attributes are defined"""

        # `getmembers_static` with custom predicate
        if missing := inspect.getmembers_static(cls, lambda val: val is _MISSING_):
            for attr_name, _ in missing:
                raise AttributeError(
                    f"`{cls.__name__}` does not define required mixin variable: `{attr_name}`"
                )
