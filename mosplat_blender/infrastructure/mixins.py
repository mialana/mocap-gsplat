from __future__ import annotations

from typing import ClassVar, Type
import logging
import inspect

from .logs import MosplatLoggingBase
from .constants import _MISSING_

class MosplatLogClassMixin:
    logger: ClassVar[logging.Logger] = _MISSING_

    @classmethod
    def create_logger_for_class(cls):
        cls.logger = MosplatLoggingBase.configure_logger_instance(
            f"{cls.__module__}.{cls.__qualname__}"
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.logger is _MISSING_:
            self.__class__.create_logger_for_class()


class MosplatBlMetaMixin(MosplatLogClassMixin):
    """A mixin class that standardizes common metadata for Blender types."""

    short_name: ClassVar[str] = _MISSING_
    prefix_suffix: ClassVar[str] = _MISSING_

    bl_category = "MOSPLAT"

    def __init_subclass__(cls, **kwargs):
        """validate and set required attributes"""
        super().__init_subclass__(**kwargs)

        cls.bl_idname = f"{cls.bl_category}_{cls.prefix_suffix}_{cls.short_name}"
        cls.bl_label = cls.bl_idname.replace("_", " ")

    @classmethod
    def at_registration(cls):
        """
        ran at registration time of the class.
        useful for differentiating 'abstract' base class behavior from registered classes
        """

        cls.create_logger_for_class()
        cls._enforce_mixin_attributes()

    @classmethod
    def _enforce_mixin_attributes(cls):
        """enforce all required attributes are defined"""

        # `getmembers_static` with custom predicate
        if missing := inspect.getmembers_static(cls, lambda val: val is _MISSING_):
            for attr_name, _ in missing:
                cls.logger.warning(
                    f"`{cls.__name__}` does not define required mixin variable: `{attr_name}`"
                )


class MosplatBlMetaPanelMixin(MosplatBlMetaMixin):
    parent_class: ClassVar[Type[MosplatBlMetaPanelMixin] | None] = _MISSING_

    @classmethod
    def at_registration(cls):
        """automate creation of `bl_parent_id` for panels."""

        super().at_registration()

        if (
            (
                cls.parent_class is not None
            )  # setting explicitly to `None` is distinct from sentinel `_MISSING_` value
            and hasattr(
                cls.parent_class, "bl_idname"
            )  # this should never be false as it is set in `__init_subclass``
            and not hasattr(
                cls, "bl_parent_id"
            )  # do not run if it already has the attribute
        ):
            cls.bl_parent_id = cls.parent_class.bl_idname
