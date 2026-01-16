"""
functions here perform runtime type-checking and raise appropriate errors
if items in the Blender ecosystem are not as expected.

if they do successfuly pass though, they perform static casting so that
we can operate with type-awareness in development.
"""

import bpy
from bpy.types import Preferences, Scene, Context

from typing import Union, cast, TYPE_CHECKING, Any, TypeAlias

from ..infrastructure.constants import ADDON_PREFERENCES_ID, ADDON_PROPERTIES_ATTRIBNAME
from ..interfaces import MosplatLoggingInterface

if TYPE_CHECKING:
    from .preferences import Mosplat_AP_Global
    from .properties import Mosplat_PG_Global
else:
    Mosplat_AP_Global: TypeAlias = Any
    Mosplat_PG_Global: TypeAlias = Any

# the only logs made here should be exceptions, which include the stack trace.
logger = MosplatLoggingInterface.configure_logger_instance(__name__)


def check_propertygroup(scene: Union[Scene, None]) -> Mosplat_PG_Global:
    if scene is None:
        raise RuntimeError("Blender scene unavailable in this context.")
    try:
        found_properties = getattr(scene, ADDON_PROPERTIES_ATTRIBNAME)
        # OK to use `cast` here as we've guarded its existence with a try-block, and we created it
        properties = cast(Mosplat_PG_Global, found_properties)
        return properties
    except AttributeError:
        raise RuntimeError(
            "Registration of addon properties was never successful. Cannot continue."
        )


def check_addonpreferences(prefs_ctx: Union[Preferences, None]) -> Mosplat_AP_Global:
    if prefs_ctx is None:
        raise RuntimeError("Blender preferences unavailable in this context.")

    try:
        found_addon = prefs_ctx.addons[ADDON_PREFERENCES_ID]
        # OK to use `cast` here as we've guarded its existence with a try-block, and we created it
        preferences = cast(Mosplat_AP_Global, found_addon.preferences)
        return preferences
    except KeyError:
        raise RuntimeError(
            "Registration of addon preferences was never successful. Cannot continue."
        )


def check_props_safe(context: Context) -> Union[Mosplat_PG_Global, None]:
    """provide a safe, non-throwing check that will log the stack trace but not raise"""
    try:
        return check_propertygroup(context.scene)
    except RuntimeError:
        return None  # log stack trace but do not raise


def check_prefs_safe(context: Context) -> Union[Mosplat_AP_Global, None]:
    """provide a safe, non-throwing check that will log the stack trace but not raise"""
    try:
        return check_addonpreferences(context.preferences)
    except RuntimeError:
        return None  # log stack trace but do not raise
