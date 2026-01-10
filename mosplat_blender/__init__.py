import bpy

from typing import Type, Sequence

from .properties import MosplatProperties
from .operators import Mosplat_OT_Base, MosplatOperatorMixin, all_operators
from .utilities import init_logging


classes: Sequence[Type[bpy.types.PropertyGroup] | Type[Mosplat_OT_Base]] = [
    MosplatProperties
] + all_operators

logger, stdout_log_handler, json_log_handler = init_logging()


def register():
    for c in classes:
        if issubclass(c, MosplatOperatorMixin):
            c.setup_bl_info()  # call a classmethod on all Mosplat operators
        bpy.utils.register_class(c)

    setattr(
        bpy.types.Scene,
        "mosplat_properties",
        bpy.props.PointerProperty(type=MosplatProperties),
    )

    logger.info("Mosplat Blender addon registration completed.")


def unregister():
    # unregister all classes
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError as e:
            logger.error(f"Error during unregistration of `{cls.__name__}`: {e}")

    try:
        delattr(bpy.types.Scene, "mosplat_properties")
    except AttributeError as e:
        logger.error(
            f"Error during unregistration of `bpy.types.Scene.mosplat_properties`: {e}"
        )

    logger.info("Mosplat Blender addon unregistration completed.")

    # remove handlers from logger
    logger.removeHandler(stdout_log_handler)
    if json_log_handler:
        logger.removeHandler(json_log_handler)
