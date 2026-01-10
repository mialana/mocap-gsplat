import bpy
import logging
from typing import Type, Sequence

from .properties import MosplatProperties
from .operators import Mosplat_OT_Base, MosplatOperatorMixin, all_operators
from .constants import LOGGER_NAME


classes: Sequence[Type[bpy.types.PropertyGroup] | Type[Mosplat_OT_Base]] = [
    MosplatProperties
] + all_operators

logger = logging.getLogger(LOGGER_NAME)


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

    # run logging operator after file load
    bpy.app.handlers.load_post.append(getattr(bpy.ops, "mosplat.logging"))


def unregister():
    global logger

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
