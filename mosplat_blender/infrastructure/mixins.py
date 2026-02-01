"""native mixin classes. note no import of `bpy`"""

from __future__ import annotations

import os
from typing import (
    ClassVar,
    Type,
    TypeVar,
    Generic,
    TYPE_CHECKING,
    List,
    NamedTuple,
    Optional,
    Callable,
    Union,
    TypeAlias,
)
from dataclasses import fields
import logging
import inspect
from functools import partial


from .protocols import SupportsCollectionProperty, SupportsDataclass
from .constants import _MISSING_
from .schemas import DeveloperError

S = TypeVar("S")

D = TypeVar("D", bound=SupportsDataclass)  # dataclass equivalent of the property group
ChildD = TypeVar("ChildD", bound=SupportsDataclass)  # dataclass that is a property
M = TypeVar("M", bound=SupportsDataclass)  # dataclass containing metadata for  class

PreregristrationFn: TypeAlias = Callable[[], None]


class CtxPackage(NamedTuple):
    context: Context
    prefs: Mosplat_AP_Global
    props: Mosplat_PG_Global


if TYPE_CHECKING:
    from bpy.types import Context
    from ..core.preferences import Mosplat_AP_Global
    from ..core.properties import Mosplat_PG_Global


class LogClassMixin:
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
            f"{cls.__module__}.logclass{label if label else cls.__name__}"
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


class EnforceAttributesMixin(Generic[M], LogClassMixin):
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

    from bpy import types

    RegistrationClasses: TypeAlias = Union[
        types.Panel,
        types.UIList,
        types.Operator,
        types.PropertyGroup,
        types.AddonPreferences,
    ]

    @classmethod
    def preregistration(cls, **kwargs):
        """
        ran at registration time of the class.
        """
        if "MOSPLAT_TESTING" in os.environ:
            cls._enforce_mixin_attributes()

    @classmethod
    def _enforce_mixin_attributes(cls):
        """enforce all required attributes are defined"""

        # `getmembers_static` with custom predicate
        missing = inspect.getmembers_static(cls, lambda val: val is _MISSING_)
        for attrname, _ in missing:
            raise DeveloperError(
                f"'{cls.__name__}' does not define required mixin variable: '{attrname}'"
            )

    @classmethod
    def preregistration_fn_factory(cls, **kwargs) -> PreregristrationFn:
        return cls.preregistration


class MetadataProxyMixin(Generic[M], EnforceAttributesMixin[M]):
    """
    this class allows to define metadata through a proxy dataclass,
    and will set the correct values before registration with Blender.
    helps to enforce required members and types
    """

    _metadata_proxy_instance: M = _MISSING_

    @classmethod
    def preregistration(cls, *, metadata: M, **kwargs):
        # iterate through fields in dataclass, set corresponding class attribute.
        for fld in fields(metadata):
            value = getattr(metadata, fld.name)
            setattr(cls, fld.name, value)

        cls._metadata_proxy_instance = metadata

        super().preregistration()

    @classmethod
    def preregistration_fn_factory(cls, *, metadata: M, **kwargs) -> PreregristrationFn:
        return partial(cls.preregistration, metadata=metadata)


class DataclassInteropMixin(Generic[D]):
    """
    a mixin that automates the transformation to/from a dataclass.
    used for Blender `PropertyGroup` classes.
    """

    # the dataclass that is synonymous to the property group.
    # will be enforced before registration.
    __dataclass_type__: Type[D] = _MISSING_

    def to_dataclass(self) -> D:
        cls = self.__dataclass_type__

        kwargs = {}
        for fld in fields(cls):
            value = getattr(self, fld.name)
            if isinstance(value, DataclassInteropMixin):
                value = value.to_dataclass()
            elif isinstance(value, SupportsCollectionProperty):
                value = self.collection_property_to_dataclass_list(value)
            kwargs[fld.name] = value

        return cls(**kwargs)

    def from_dataclass(self, dc: D) -> None:
        for fld in fields(dc):
            value = getattr(dc, fld.name)
            target = getattr(self, fld.name, None)
            if isinstance(target, DataclassInteropMixin):
                target.from_dataclass(value)
            elif isinstance(target, SupportsCollectionProperty):
                target.clear()  # clear out the old collection property
                for item_dc in value:
                    item_pg = target.add()
                    if isinstance(item_pg, DataclassInteropMixin):
                        item_pg.from_dataclass(item_dc)
                    else:
                        item_pg = item_dc
            else:
                setattr(self, fld.name, value)

    @staticmethod
    def collection_property_to_dataclass_list(
        cp: SupportsCollectionProperty[DataclassInteropMixin[ChildD]],
    ) -> List[ChildD]:
        return [v.to_dataclass() for v in cp]


class APAccessorMixin:
    """a mixin class for any class that wants local access to global preferences"""

    from bpy.types import Context  # local import

    if TYPE_CHECKING:
        from ..core.preferences import Mosplat_AP_Global

    @staticmethod
    def prefs(context) -> Mosplat_AP_Global:
        from ..core.checks import check_addonpreferences

        return check_addonpreferences(context.preferences)


class PGAccessorMixin:
    """a mixin class for any class that wants local access to global properties"""

    from bpy.types import Context  # local import

    if TYPE_CHECKING:
        from ..core.properties import Mosplat_PG_Global

    @staticmethod
    def props(context: Context) -> Mosplat_PG_Global:
        from ..core.checks import check_propertygroup

        return check_propertygroup(context.scene)


class BlRNAAccessorMixin(EnforceAttributesMixin):
    """a mixin class for easier access to Blender properties' RNA"""

    from bpy.types import BlenderRNA

    bl_rna: ClassVar[BlenderRNA]

    @classmethod
    def get_prop_name(cls, attrname: str) -> str:
        try:
            property = cls.bl_rna.properties[attrname]
            return property.name
        except KeyError:
            msg = f"Property '{attrname}' does not exist on '{cls.__name__}'"
            cls.class_logger.error(msg)
            return msg

    @classmethod
    def get_prop_id(cls, attrname: str) -> str:
        try:
            property = cls.bl_rna.properties[attrname]
            return property.identifier
        except KeyError:
            msg = f"Property '{attrname}' does not exist on '{cls.__name__}'"
            cls.class_logger.error(msg)
            return msg


class ContextAccessorMixin(
    Generic[M],
    MetadataProxyMixin[M],
    APAccessorMixin,
    PGAccessorMixin,
):

    if TYPE_CHECKING:
        from bpy.types import Context

    @classmethod
    def package(cls, context: Context) -> CtxPackage:
        """convenience method to package context"""
        return CtxPackage(
            context=context, prefs=cls.prefs(context), props=cls.props(context)
        )
