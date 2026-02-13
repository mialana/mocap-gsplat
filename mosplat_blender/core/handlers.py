"""methods that are hooked by `bpy.app.handlers`"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from bpy.app.handlers import persistent

from ..infrastructure.schemas import MediaIOMetadata, UserFacingError
from ..interfaces.logging_interface import LoggingInterface
from .checks import check_addonpreferences, check_propertygroup

if TYPE_CHECKING:
    from bpy.types import Scene

    from .preferences import Mosplat_AP_Global
    from .properties import Mosplat_PG_Global

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

    data, msg = load_metadata_dataclass_from_json(props, prefs)
    props.metadata_accessor.from_dataclass(data)  # transfer data to property group
    return data, msg


def load_metadata_dataclass_from_json(
    props: Mosplat_PG_Global, prefs: Mosplat_AP_Global
) -> Tuple[MediaIOMetadata, str]:
    """base entrypoint for restoring to dataclass"""
    media_directory = props.media_directory_
    dc: MediaIOMetadata = MediaIOMetadata(base_directory=str(media_directory))

    try:

        # get destination path for json
        json_filepath = props.media_io_metadata_filepath(prefs)

        load_msg = dc.load_from_JSON(json_path=json_filepath)

        return (dc, load_msg)
    except (UserWarning, UserFacingError) as e:
        msg = UserFacingError.make_msg(
            "Unable to parse media directory for metadata. We can fall back to default values.",
            e,
        )
        return (dc, msg)


@persistent
def handle_save_to_json(scene: Scene):
    from bpy import context

    props = check_propertygroup(scene)
    prefs = check_addonpreferences(context.preferences)

    props.metadata_accessor.to_JSON(props.media_io_metadata_filepath(prefs))


@persistent
def handle_reset_properties(scene: Scene):
    handle_reset_properties_timer_entrypoint(scene)


def handle_reset_properties_timer_entrypoint(scene: Optional[Scene] = None) -> None:
    from bpy import context

    scene = scene or context.scene
    props = check_propertygroup(scene)

    meta = props._meta

    for prop_meta in meta:
        props.property_unset(prop_meta.id)

    logger.info("Properties reset.")


@persistent
def handle_set_render_engine(scene: Scene):
    pass


def handle_set_render_engine_timer_entrypoint(scene: Optional[Scene] = None) -> None:
    from bpy import context

    scene = scene or context.scene
    if not scene:
        return

    try:
        setattr(scene.render, "engine", "CYCLES")
        setattr(scene.cycles, "device", "GPU")
    except AttributeError as e:
        msg = UserFacingError.make_msg(
            "Could not configure render engine as 'CYCLES'. Try again manually in UI.",
            e,
        )
        logger.error(msg)
