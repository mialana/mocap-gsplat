"""
main entrypoint for addon.
moves implementation logic and imports out of `__init__.py`.
"""

import bpy

from typing import Type, Sequence, Union
from functools import partial

from . import core
from .interfaces import MosplatLoggingInterface, MosplatVGGTInterface
from .infrastructure.mixins import MosplatEnforceAttributesMixin
from .core.checks import check_addonpreferences
from .core.handlers import (
    handle_restore_from_json,
    handle_restore_from_json_timer_entrypoint,
)
from .infrastructure.constants import ADDON_PROPERTIES_ATTRIBNAME, ADDON_HUMAN_READABLE

classes: Sequence[
    Union[
        Type[core.MosplatOperatorBase],
        Type[core.MosplatPanelBase],
        Type[core.MosplatPropertyGroupBase],
        Type[core.Mosplat_AP_Global],
    ]
] = (
    [
        core.Mosplat_AP_Global,
    ]
    + core.all_properties
    + core.all_operators
    + core.all_panels
)

logger = MosplatLoggingInterface.configure_logger_instance(__name__)


def register_addon():
    for c in classes:
        try:
            if issubclass(c, MosplatEnforceAttributesMixin):
                c.at_registration()  # do any necessary class-level changes
            bpy.utils.register_class(c)
        except (RuntimeError, AttributeError):
            logger.exception(f"Exception during registration: `{c.__name__=}`")

    setattr(
        bpy.types.Scene,
        ADDON_PROPERTIES_ATTRIBNAME,
        bpy.props.PointerProperty(type=core.Mosplat_PG_Global),
    )

    # do not catch thrown exceptions as we should not successfully register without addon preferences
    addon_preferences: core.Mosplat_AP_Global = check_addonpreferences(
        bpy.context.preferences
    )

    MosplatLoggingInterface.init_handlers_from_addon_prefs(addon_preferences)

    # load from JSON both every file load and
    bpy.app.handlers.load_post.append(handle_restore_from_json)
    bpy.app.timers.register(handle_restore_from_json_timer_entrypoint, first_interval=0)

    logger.info(f"'{ADDON_HUMAN_READABLE}' addon registration completed.")


def unregister_addon():
    """essentially all operations here should be guarded with try blocks"""
    # unregister all classes
    for c in reversed(classes):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            logger.exception(f"Exception during unregistration of `{c.__name__=}`")

    try:
        delattr(bpy.types.Scene, ADDON_PROPERTIES_ATTRIBNAME)
    except AttributeError:
        logger.exception(f"Error during unregistration of add-on properties")

    try:
        MosplatVGGTInterface.cleanup()
    except Exception:
        logger.exception(f"Error while cleaning up VGGT interface")

    bpy.app.handlers.load_post.remove(handle_restore_from_json)

    logger.info(f"'{ADDON_HUMAN_READABLE}' addon unregistration completed.")
