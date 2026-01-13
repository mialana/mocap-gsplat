"""
main entrypoint for addon.
moves implementation logic and imports out of `__init__.py`.
"""

import bpy

from typing import Type, Sequence, cast

from . import core
from .interfaces import MosplatLoggingInterface
from .infrastructure.mixins import MosplatBlMetaMixin
from .infrastructure.checks import check_addonpreferences
from .infrastructure.constants import ADDON_ID, ADDON_PROPERTIES_ATTRIBNAME

classes: Sequence[
    Type[bpy.types.PropertyGroup]
    | Type[bpy.types.AddonPreferences]
    | Type[core.MosplatOperatorBase]
    | Type[core.MosplatPanelBase]
] = (
    [core.Mosplat_PG_Global, core.Mosplat_AP_Global]
    + core.all_operators
    + core.all_panels
)

logger = MosplatLoggingInterface.configure_logger_instance(__name__)


def register_addon():
    for c in classes:
        try:
            if issubclass(c, MosplatBlMetaMixin):
                c.at_registration()  # do any necessary class-level changes
            bpy.utils.register_class(c)
        except Exception:
            logger.exception(f"Exception during registration: `{c.__name__=}`")

        setattr(
            bpy.types.Scene,
            ADDON_PROPERTIES_ATTRIBNAME,
            bpy.props.PointerProperty(type=core.Mosplat_PG_Global),
        )

    # do not catch thrown exceptions as we should not successfully register without addon preferences
    addon_preferences = check_addonpreferences(bpy.context.preferences)
    MosplatLoggingInterface.init_handlers_from_addon_prefs(addon_preferences)

    logger.info("Mosplat Blender addon registration completed.")


def unregister_addon():
    # unregister all classes
    for c in reversed(classes):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            logger.exception(f"Exception during unregistration of `{c.__name__=}`")

    try:
        delattr(bpy.types.Scene, ADDON_PROPERTIES_ATTRIBNAME)
    except AttributeError:
        logger.exception(f"Error during unregistration of add-on properties")

    logger.info("Mosplat Blender addon unregistration completed.")

    MosplatLoggingInterface.cleanup()
