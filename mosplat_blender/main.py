import bpy

from typing import Type, Sequence
from pathlib import Path

from .properties import Mosplat_Properties
from .preferences import Mosplat_AddonPreferences
from .operators import all_operators
from .panels import all_panels
from .infrastructure.logs import MosplatLoggingManager
from .infrastructure.bases import Mosplat_OT_Base, Mosplat_PT_Base
from .infrastructure.mixins import MosplatBlMetaMixin

classes: Sequence[
    Type[bpy.types.PropertyGroup]
    | Type[bpy.types.AddonPreferences]
    | Type[Mosplat_OT_Base]
    | Type[Mosplat_PT_Base]
] = ([Mosplat_Properties, Mosplat_AddonPreferences] + all_operators + all_panels)

logger = MosplatLoggingManager.configure_logger_instance(__name__)


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
            "mosplat_properties",
            bpy.props.PointerProperty(type=Mosplat_Properties),
        )
    try:
        MosplatLoggingManager.init_handlers_from_addon_prefs()
    except Exception:
        logger.exception(
            "Something went wrong when initializing handlers. Continuing registration."
        )  # just in case something goes wrong, this should not prevent registration

    logger.info("Mosplat Blender addon registration completed.")


def unregister_addon():
    # unregister all classes
    for c in reversed(classes):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            logger.exception(f"Exception during unregistration of `{c.__name__=}`")

    try:
        delattr(bpy.types.Scene, "mosplat_properties")
    except AttributeError:
        logger.exception(f"Error during unregistration of add-on properties")

    logger.info("Mosplat Blender addon unregistration completed.")

    MosplatLoggingManager.cleanup()
