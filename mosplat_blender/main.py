"""
main entrypoint for addon.
moves implementation logic and imports out of `__init__.py`.
"""

from typing import Sequence, Tuple, Type, Union

import bpy

from . import core
from .core.checks import check_addonpreferences
from .core.handlers import (
    handle_load_from_json,
    handle_load_from_json_timer_entrypoint,
    handle_save_to_json,
)
from .infrastructure.constants import ADDON_GLOBAL_PROPS_NAME, ADDON_HUMAN_READABLE
from .infrastructure.mixins import PreregristrationFn
from .infrastructure.schemas import DeveloperError, UnexpectedError
from .interfaces import MosplatLoggingInterface

logger = MosplatLoggingInterface.configure_logger_instance(__name__)

registration_factory: Sequence[
    Tuple[
        Union[
            Type[core.MosplatOperatorBase],
            Type[core.MosplatPanelBase],
            Type[core.MosplatUIListBase],
            Type[core.Mosplat_AP_Global],
            Type[core.MosplatPropertyGroupBase],
        ],
        PreregristrationFn,
    ],
] = (
    core.operator_factory
    + core.panel_factory
    + core.ui_list_factory
    + [core.preferences_factory]  # addon preferences is a singleton class
    + core.properties_factory
)


def register_addon():
    bpy.utils.register_classes_factory
    for cls, pregistration_fn in registration_factory:
        try:
            pregistration_fn()  # we call pre-registration function here
            bpy.utils.register_class(cls)
        except (ValueError, RuntimeError, AttributeError) as e:
            raise DeveloperError(
                f"Exception during registration of `{cls.__name__}`.", e
            ) from e

    # do not catch thrown exceptions as we should not successfully register without addon preferences or property groups
    setattr(
        bpy.types.Scene,
        ADDON_GLOBAL_PROPS_NAME,
        bpy.props.PointerProperty(type=core.Mosplat_PG_Global),
    )

    addon_preferences: core.Mosplat_AP_Global = check_addonpreferences(
        bpy.context.preferences
    )

    MosplatLoggingInterface()._init_handlers_from_addon_prefs(addon_preferences)

    # try load from JSON every file load and after registration occurs
    bpy.app.handlers.load_post.append(handle_load_from_json)
    bpy.app.timers.register(handle_load_from_json_timer_entrypoint, first_interval=0)

    bpy.app.handlers.undo_post.append(handle_save_to_json)
    bpy.app.handlers.redo_post.append(handle_save_to_json)

    logger.info(f"'{ADDON_HUMAN_READABLE}' addon registration completed.")


def unregister_addon():
    """essentially all operations here should be guarded with try blocks"""
    # unregister all classes
    for cls, _ in reversed(registration_factory):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            logger.error(f"Exception during unregistration of `{cls.__name__=}`")

    try:
        delattr(bpy.types.Scene, ADDON_GLOBAL_PROPS_NAME)
    except AttributeError:
        logger.error(f"Error removing add-on properties.")

    try:
        from .interfaces import MosplatVGGTInterface

        MosplatVGGTInterface.cleanup_interface()
    except UnexpectedError as e:
        logger.error(f"Error during VGGT cleanup: {str(e)}")

    bpy.app.handlers.load_post.remove(handle_load_from_json)

    logger.info(f"'{ADDON_HUMAN_READABLE}' addon unregistration completed.")
