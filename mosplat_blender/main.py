import bpy

from os import environ
from typing import Type, Sequence
from pathlib import Path
from dotenv import load_dotenv

from .properties import MosplatProperties
from .operators import all_operators
from .panels import all_panels
from .infrastructure.logs import MosplatLoggingBase
from .infrastructure.bases import Mosplat_OT_Base, Mosplat_PT_Base
from .infrastructure.mixins import MosplatBlMetaMixin


classes: Sequence[
    Type[bpy.types.PropertyGroup] | Type[Mosplat_OT_Base] | Type[Mosplat_PT_Base]
] = ([MosplatProperties] + all_operators + all_panels)

logger = MosplatLoggingBase.configure_logger_instance(__name__)


def register_addon():
    dotenv_path = Path(__file__).resolve().parent.joinpath(".env")
    load_dotenv(
        dotenv_path, verbose=True, override=True
    )  # load all env variables from local `.env`
    logger.info("Environment variables loaded.")

    for c in classes:
        try:
            if issubclass(c, MosplatBlMetaMixin):
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


def unregister_addon():
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

    MosplatLoggingBase.cleanup()
