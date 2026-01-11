import bpy

from typing import Type, Sequence

from .properties import MosplatProperties
from .operators import all_operators
from .panels import all_panels
from .utilities import MosplatLogging
from .types import MosplatBlBaseMixin, Mosplat_OT_Base, Mosplat_PT_Base


classes: Sequence[
    Type[bpy.types.PropertyGroup] | Type[Mosplat_OT_Base] | Type[Mosplat_PT_Base]
] = ([MosplatProperties] + all_operators + all_panels)

logger = MosplatLogging.default_logger()


def register():
    for c in classes:
        try:
            if issubclass(c, MosplatBlBaseMixin):
                c.at_registration()  # do any necessary class-level changes
            bpy.utils.register_class(c)
        except Exception as e:
            logger.exception(f"Error during registration: `{c.__name__=}`")

    setattr(
        bpy.types.Scene,
        "mosplat_properties",
        bpy.props.PointerProperty(type=MosplatProperties),
    )

    logger.info("Mosplat Blender addon registration completed.")


def unregister():
    # unregister all classes
    for c in reversed(classes):
        try:
            bpy.utils.unregister_class(c)
        except Exception as e:
            logger.warning(
                f"Exception occured during unregistration of `{c.__name__}`: {e}"
            )

    try:
        delattr(bpy.types.Scene, "mosplat_properties")
    except AttributeError as e:
        logger.error(
            f"Error during unregistration of `bpy.types.Scene.mosplat_properties`: {e}"
        )

    logger.info("Mosplat Blender addon unregistration completed.")

    MosplatLogging.cleanup()
