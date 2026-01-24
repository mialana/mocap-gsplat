"""methods that are hooked by `bpy.app.handlers`"""

import bpy
from bpy.types import Scene
from bpy.app.handlers import persistent

from typing import TYPE_CHECKING, TypeAlias, TypeAlias, Any, Optional

from .checks import check_propertygroup, check_addonpreferences

from ..infrastructure.schemas import MediaIOMetadata
from ..interfaces.logging_interface import MosplatLoggingInterface

if TYPE_CHECKING:
    from .properties import Mosplat_PG_Global
    from .preferences import Mosplat_AP_Global
else:
    Mosplat_PG_Global: TypeAlias = Any
    Mosplat_AP_Global: TypeAlias = Any

logger = MosplatLoggingInterface.configure_logger_instance(__name__)


@persistent
def handle_restore_from_json(scene: Scene):
    """entrypoint for `bpy.app.handlers.load_post`"""
    props = check_propertygroup(scene)

    restore_metadata_from_json(props)


def handle_restore_from_json_timer_entrypoint():
    """entrypoint for `bpy.app.timers`"""
    return handle_restore_from_json(bpy.context.scene)


def restore_metadata_from_json(
    props: Mosplat_PG_Global, prefs: Optional[Mosplat_AP_Global] = None
):
    """base entrypoint"""
    prefs = prefs or check_addonpreferences(bpy.context.preferences)
    metadata_prop = props.metadata

    json_filepath = props.metadata_json_filepath(prefs)

    if not json_filepath.exists:
        logger.info("No JSON file to be restored.")

    dc = MediaIOMetadata.from_JSON(json_filepath)
    metadata_prop.from_dataclass(dc)

    logger.info("Metadata JSON file successfully restored.")
