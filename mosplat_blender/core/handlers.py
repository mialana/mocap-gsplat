"""methods that are hooked by `bpy.app.handlers`"""

import bpy
from bpy.types import Scene
from bpy.app.handlers import persistent

from typing import Optional

from .checks import check_propertygroup, check_addonpreferences, check_json_filepath
from .properties import Mosplat_PG_MediaIOMetadata
from ..infrastructure.schemas import MediaIOMetadata


@persistent
def handle_restore_from_json(scene: Optional[Scene]):

    props = check_propertygroup(scene)
    prefs = check_addonpreferences(bpy.context.preferences)
    metadata_prop: Mosplat_PG_MediaIOMetadata = props.current_media_io_metadata

    json_filepath = check_json_filepath(prefs, props)

    data = MediaIOMetadata.from_JSON(json_filepath)
    metadata_prop.from_dataclass(data)


def handle_restore_from_json_timer_entrypoint():

    return handle_restore_from_json(bpy.context.scene)
