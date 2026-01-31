"""methods that are hooked by `bpy.app.handlers`"""

from __future__ import annotations

import bpy
from bpy.types import Scene
from bpy.app.handlers import persistent

from typing import TYPE_CHECKING, Optional, Tuple

from .checks import check_propertygroup, check_addonpreferences

from ..infrastructure.schemas import MediaIODataset
from ..interfaces.logging_interface import MosplatLoggingInterface

if TYPE_CHECKING:
    from .properties import Mosplat_PG_Global
    from .preferences import Mosplat_AP_Global

logger = MosplatLoggingInterface.configure_logger_instance(__name__)


@persistent
def handle_load_from_json(scene: Scene):
    """entrypoint for `bpy.app.handlers.load_post`"""

    bpy.app.timers.register(
        lambda: handle_load_from_json_timer_entrypoint(scene),
        first_interval=0,
    )


def handle_load_from_json_timer_entrypoint(scene: Optional[Scene] = None) -> None:
    """entrypoint for `bpy.app.timers`"""
    scene = scene or bpy.context.scene
    props = check_propertygroup(scene)

    try:
        # don't need dataclass anymore in this entrypoint pathway
        _, load_msg = load_dataset_property_group_from_json(props)
        logger.info(load_msg)  # log the load msg
    except UserWarning as e:
        # warn but there's not much that can be done about permission issues
        # try to proceed
        logger.warning(str(e))


def load_dataset_property_group_from_json(
    props: Mosplat_PG_Global, prefs: Optional[Mosplat_AP_Global] = None
) -> Tuple[MediaIODataset, str]:
    """base entrypoint for restoring directly to property group"""
    prefs = prefs or check_addonpreferences(bpy.context.preferences)

    result = load_dataset_dataclass_from_json(props, prefs)
    props.dataset_accessor.from_dataclass(result[0])  # transfer data to property group

    return result


def load_dataset_dataclass_from_json(
    props: Mosplat_PG_Global, prefs: Mosplat_AP_Global
) -> Tuple[MediaIODataset, str]:
    """base entrypoint for restoring to dataclass"""
    current_media_dirpath = props.current_media_dirpath
    dc = MediaIODataset(base_directory=str(current_media_dirpath))

    # get destination path for json
    json_filepath = props.data_json_filepath(prefs)

    load_msg = dc.load_from_JSON(json_path=json_filepath)

    return (dc, load_msg)


@persistent
def handle_save_to_json(scene: Scene):
    props = check_propertygroup(scene)
    prefs = check_addonpreferences(bpy.context.preferences)

    props.dataset_accessor.to_JSON(props.data_json_filepath(prefs))
