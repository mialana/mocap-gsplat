"""
functions here perform runtime type-checking and raise appropriate errors
if items in the Blender ecosystem are not as expected.

if they do successfuly pass though, they perform static casting so that
we can operate with type-awareness in development.
"""

from bpy.types import Preferences, Scene, Context

from typing import Union, cast, TYPE_CHECKING, Any, TypeAlias, NoReturn, Optional

from ..infrastructure.constants import ADDON_PREFERENCES_ID, ADDON_PROPERTIES_ATTRIBNAME
from ..infrastructure.schemas import UnexpectedError, SafeError
from ..interfaces import MosplatLoggingInterface

if TYPE_CHECKING:
    from .preferences import Mosplat_AP_Global
    from .properties import Mosplat_PG_Global
else:
    Mosplat_AP_Global: TypeAlias = Any
    Mosplat_PG_Global: TypeAlias = Any

logger = MosplatLoggingInterface.configure_logger_instance(__name__)


def check_propertygroup(
    scene: Optional[Scene],
) -> Union[Mosplat_PG_Global, NoReturn]:
    if scene is None:  # can occur if checked at the wrong time
        raise SafeError("Blender scene unavailable in this context.")
    try:
        found_properties = getattr(scene, ADDON_PROPERTIES_ATTRIBNAME)
        # OK to use `cast` here as we've guarded its existence with a try-block, and we created it
        properties = cast(Mosplat_PG_Global, found_properties)
        return properties
    except AttributeError as e:
        raise UnexpectedError(
            "Registration of addon properties was never successful. Cannot continue."
        ) from e


def check_addonpreferences(
    prefs_ctx: Optional[Preferences],
) -> Union[Mosplat_AP_Global, NoReturn]:
    if prefs_ctx is None:  # can occur if checked at the wrong time
        raise SafeError("Blender preferences unavailable in this context.")

    try:
        found_addon = prefs_ctx.addons[ADDON_PREFERENCES_ID]
        # OK to use `cast` here as we've guarded its existence with a try-block, and we created it
        preferences = cast(Mosplat_AP_Global, found_addon.preferences)
        return preferences
    except KeyError as e:
        raise UnexpectedError(
            "Registration of addon preferences was never successful. Cannot continue."
        ) from e


def check_props_safe(context: Context) -> Optional[Mosplat_PG_Global]:
    """provide a safe, non-throwing check that will log the stack trace but not raise"""
    try:
        return check_propertygroup(context.scene)
    except SafeError as e:
        logger.exception(str(e))
        return None  # log stack trace but do not raise


def check_prefs_safe(context: Context) -> Optional[Mosplat_AP_Global]:
    """provide a safe, non-throwing check that will log the stack trace but not raise"""
    try:
        return check_addonpreferences(context.preferences)
    except SafeError as e:
        logger.exception(str(e))
        return None  # log stack trace but do not raise
