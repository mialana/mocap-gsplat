"""
main entrypoint for addon.
moves implementation logic and imports out of `__init__.py`.
"""

from typing import Sequence, Tuple, Type, Union

import bpy

import core
from infrastructure.mixins import PreregristrationFn
from infrastructure.schemas import AddonMeta, DeveloperError, UnexpectedError
from interfaces import LoggingInterface
from operators import MosplatOperatorBase, operator_factory

logger = LoggingInterface.configure_logger_instance(__name__)

registration_factory: Sequence[
    Tuple[
        Union[
            Type[MosplatOperatorBase],
            Type[core.MosplatPanelBase],
            Type[core.MosplatUIListBase],
            Type[core.Mosplat_AP_Global],
            Type[core.MosplatPropertyGroupBase],
        ],
        PreregristrationFn,
    ],
] = (
    operator_factory
    + core.panel_factory
    + core.ui_list_factory
    + core.properties_factory
    + [core.preferences_factory]  # addon preferences is a singleton class
)

ADDON_META = AddonMeta()


def register_addon():
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
        ADDON_META.global_props_name,
        bpy.props.PointerProperty(type=core.Mosplat_PG_Global),
    )

    addon_preferences: core.Mosplat_AP_Global = core.checks.check_addonpreferences(
        bpy.context.preferences
    )

    LoggingInterface().init_handlers_from_addon_prefs(addon_preferences)

    set_handlers()

    logger.info(f"'{ADDON_META.human_readable_name}' addon registration completed.")


def unregister_addon():
    """essentially all operations here should be guarded with try blocks"""
    try:
        delattr(bpy.types.Scene, ADDON_META.global_props_name)
    except AttributeError:
        logger.error(f"Error removing add-on properties.")

    # unregister all classes
    for cls, _ in reversed(registration_factory):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            logger.error(f"Exception during unregistration of `{cls.__name__=}`")

    try:
        from interfaces import VGGTInterface

        VGGTInterface.cleanup_interface()
    except UnexpectedError as e:
        logger.error(f"Error during VGGT cleanup: {str(e)}")

    unset_handlers()

    logger.info(f"'{ADDON_META.human_readable_name}' addon unregistration completed.")


def set_handlers():
    load_post = bpy.app.handlers.load_post
    if not core.handlers.handle_load_from_json in load_post:
        load_post.append(core.handlers.handle_load_from_json)
    if not core.handlers.handle_reset_properties in load_post:
        load_post.append(core.handlers.handle_reset_properties)
    if not core.handlers.handle_set_render_engine in load_post:
        load_post.append(core.handlers.handle_set_render_engine)

    if not core.handlers.handle_save_to_json in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(core.handlers.handle_save_to_json)
    if not core.handlers.handle_save_to_json in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.append(core.handlers.handle_save_to_json)

    # register timers to run after add-on registration
    bpy.app.timers.register(
        core.handlers.handle_load_from_json_timer_entrypoint, first_interval=0
    )
    bpy.app.timers.register(
        core.handlers.handle_reset_properties_timer_entrypoint, first_interval=0
    )
    bpy.app.timers.register(
        core.handlers.handle_set_render_engine_timer_entrypoint, first_interval=0
    )


def unset_handlers():
    load_post = bpy.app.handlers.load_post
    if core.handlers.handle_load_from_json in load_post:
        load_post.remove(core.handlers.handle_load_from_json)
    if core.handlers.handle_reset_properties in load_post:
        load_post.remove(core.handlers.handle_reset_properties)
    if core.handlers.handle_set_render_engine in load_post:
        load_post.remove(core.handlers.handle_set_render_engine)
    if core.handlers.handle_save_to_json in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.remove(core.handlers.handle_save_to_json)
    if core.handlers.handle_save_to_json in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.remove(core.handlers.handle_save_to_json)
