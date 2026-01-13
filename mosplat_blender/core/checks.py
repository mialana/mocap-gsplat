"""
functions here perform runtime type-checking and raise appropriate errors
if items in the Blender ecosystem are not as expected.

if they do successfuly pass though, they perform static casting so that
we can operate with type-awareness in development.
"""

from bpy.types import Preferences
from typing import Union, cast

from ..infrastructure.constants import ADDON_ID
from . import Mosplat_AP_Global
from ..interfaces import MosplatLoggingInterface

# the only logs made here should be exceptions, which include the stack trace.
logger = MosplatLoggingInterface.configure_logger_instance(__name__)


def check_addonpreferences(prefs_ctx: Union[Preferences, None]) -> Mosplat_AP_Global:
    if prefs_ctx is None:
        raise RuntimeError("Blender preferences unavailable in this context.")

    try:
        found_addon = prefs_ctx.addons[ADDON_ID]
        # OK to use `cast` here as we've guarded its existence with a try-block, and we created it
        preferences = cast(Mosplat_AP_Global, found_addon.preferences)
        return preferences
    except KeyError:
        logger.exception(
            "Registration of addon preferences was never successful. Cannot continue."
        )
        raise
