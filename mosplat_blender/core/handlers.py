"""methods that are hooked by `bpy.app.handlers`"""

from __future__ import annotations

import bpy
from bpy.types import Scene
from bpy.app.handlers import persistent

from typing import TYPE_CHECKING, Optional

from .checks import check_propertygroup, check_addonpreferences

from ..infrastructure.schemas import MediaIODataset
from ..interfaces.logging_interface import MosplatLoggingInterface

if TYPE_CHECKING:
    from .properties import Mosplat_PG_Global
    from .preferences import Mosplat_AP_Global

logger = MosplatLoggingInterface.configure_logger_instance(__name__)


@persistent
def handle_restore_from_json(scene: Scene):
    """entrypoint for `bpy.app.handlers.load_post`"""
    props = check_propertygroup(scene)
    restore_dataset_from_json(props)


def handle_restore_from_json_timer_entrypoint():
    """entrypoint for `bpy.app.timers`"""
    return handle_restore_from_json(bpy.context.scene)


def restore_dataset_from_json(
    props: Mosplat_PG_Global, prefs: Optional[Mosplat_AP_Global] = None
):
    """base entrypoint"""
    prefs = prefs or check_addonpreferences(bpy.context.preferences)

    # get destination path for json
    json_filepath = props.data_json_filepath(prefs)

    if not json_filepath.exists:
        logger.info("No JSON file to be restored. Creating default dataset.")

    current_media_dirpath = props.current_media_dirpath

    dc = MediaIODataset.from_JSON(
        json_path=json_filepath, base_directory=current_media_dirpath
    )

    props.dataset_accessor.from_dataclass(dc)

    logger.info("Properties synced with cached dataset.")


@persistent
def handle_save_to_json(scene: Scene):
    props = check_propertygroup(scene)
    prefs = check_addonpreferences(bpy.context.preferences)

    props.dataset_accessor.to_JSON(props.data_json_filepath(prefs))
