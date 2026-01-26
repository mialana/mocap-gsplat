"""methods that are hooked by `bpy.app.handlers`"""

import bpy
from bpy.types import Scene
from bpy.app.handlers import persistent

from typing import TYPE_CHECKING, TypeAlias, TypeAlias, Any, Optional

from .checks import check_propertygroup, check_addonpreferences

from ..infrastructure.schemas import MediaIODataset, UserFacingError
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

    from .properties import Mosplat_PG_AppliedPreprocessScript
    from ..infrastructure.schemas import AppliedPreprocessScript

    script = AppliedPreprocessScript.now(str(prefs.preprocess_media_script_file))

    for range in props.dataset_accessor.ranges_accessor:
        new: Mosplat_PG_AppliedPreprocessScript = range.scripts_accessor.add()
        new.from_dataclass(script)

        new2: Mosplat_PG_AppliedPreprocessScript = range.scripts_accessor.add()
        new2.from_dataclass(script)
