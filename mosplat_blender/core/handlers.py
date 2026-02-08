"""methods that are hooked by `bpy.app.handlers`"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from bpy.app.handlers import persistent

from core.checks import check_addonpreferences, check_propertygroup
from infrastructure.schemas import MediaIOMetadata
from interfaces.logging_interface import LoggingInterface

if TYPE_CHECKING:
    from bpy.types import Scene

    from core.preferences import Mosplat_AP_Global
    from core.properties import Mosplat_PG_Global

logger = LoggingInterface.configure_logger_instance(__name__)


@persistent
def handle_load_from_json(scene: Scene):
    """entrypoint for `bpy.app.handlers.load_post`"""

    handle_load_from_json_timer_entrypoint(scene)


def handle_load_from_json_timer_entrypoint(scene: Optional[Scene] = None) -> None:
    """entrypoint for `bpy.app.timers`"""
    from bpy import context

    scene = scene or context.scene
    props = check_propertygroup(scene)

    try:
        # don't need dataclass anymore in this entrypoint pathway
        _, load_msg = load_metadata_property_group_from_json(props)
        logger.info(load_msg)  # log the load msg
    except UserWarning as e:
        # warn but there's not much that can be done about permission issues
        # try to proceed
        logger.warning(str(e))


def load_metadata_property_group_from_json(
    props: Mosplat_PG_Global, prefs: Optional[Mosplat_AP_Global] = None
) -> Tuple[MediaIOMetadata, str]:
    from bpy import context

    """base entrypoint for restoring directly to property group"""
    prefs = prefs or check_addonpreferences(context.preferences)

    result = load_metadata_dataclass_from_json(props, prefs)
    props.metadata_accessor.from_dataclass(result[0])  # transfer data to property group

    return result


def load_metadata_dataclass_from_json(
    props: Mosplat_PG_Global, prefs: Mosplat_AP_Global
) -> Tuple[MediaIOMetadata, str]:
    """base entrypoint for restoring to dataclass"""
    media_directory = props.media_directory_
    dc = MediaIOMetadata(base_directory=str(media_directory))

    # get destination path for json
    json_filepath = props.media_io_metadata_filepath(prefs)

    load_msg = dc.load_from_JSON(json_path=json_filepath)

    return (dc, load_msg)


@persistent
def handle_save_to_json(scene: Scene):
    from bpy import context

    props = check_propertygroup(scene)
    prefs = check_addonpreferences(context.preferences)

    props.metadata_accessor.to_JSON(props.media_io_metadata_filepath(prefs))


@persistent
def handle_reset_properties(scene: Scene):
    from bpy import context

    scene = scene or context.scene
    props = check_propertygroup(scene)

    meta = props._meta

    for prop_meta in meta:
        props.property_unset(prop_meta.id)

    logger.info("Properties reset.")
