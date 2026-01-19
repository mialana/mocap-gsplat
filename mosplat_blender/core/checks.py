"""
functions here perform runtime type-checking and raise appropriate errors
if items in the Blender ecosystem are not as expected.

if they do successfuly pass though, they perform static casting so that
we can operate with type-awareness in development.
"""

from bpy.types import Preferences, Scene, Context
from bpy.props import CollectionProperty

from typing import Union, cast, TYPE_CHECKING, Any, TypeAlias, NoReturn
from pathlib import Path

from ..infrastructure.constants import ADDON_PREFERENCES_ID, ADDON_PROPERTIES_ATTRIBNAME
from ..interfaces import MosplatLoggingInterface

if TYPE_CHECKING:
    from .preferences import Mosplat_AP_Global
    from .properties import Mosplat_PG_Global
else:
    Mosplat_AP_Global: TypeAlias = Any
    Mosplat_PG_Global: TypeAlias = Any

logger = MosplatLoggingInterface.configure_logger_instance(__name__)


def check_propertygroup(scene: Union[Scene, None]) -> Mosplat_PG_Global:
    if scene is None:
        raise RuntimeError("Blender scene unavailable in this context.")
    try:
        found_properties = getattr(scene, ADDON_PROPERTIES_ATTRIBNAME)
        # OK to use `cast` here as we've guarded its existence with a try-block, and we created it
        properties = cast(Mosplat_PG_Global, found_properties)
        return properties
    except AttributeError:
        raise RuntimeError(
            "Registration of addon properties was never successful. Cannot continue."
        )


def check_addonpreferences(prefs_ctx: Union[Preferences, None]) -> Mosplat_AP_Global:
    if prefs_ctx is None:
        raise RuntimeError("Blender preferences unavailable in this context.")

    try:
        found_addon = prefs_ctx.addons[ADDON_PREFERENCES_ID]
        # OK to use `cast` here as we've guarded its existence with a try-block, and we created it
        preferences = cast(Mosplat_AP_Global, found_addon.preferences)
        return preferences
    except KeyError:
        raise RuntimeError(
            "Registration of addon preferences was never successful. Cannot continue."
        )


def check_props_safe(context: Context) -> Union[Mosplat_PG_Global, None]:
    """provide a safe, non-throwing check that will log the stack trace but not raise"""
    try:
        return check_propertygroup(context.scene)
    except RuntimeError:
        return None  # log stack trace but do not raise


def check_prefs_safe(context: Context) -> Union[Mosplat_AP_Global, None]:
    """provide a safe, non-throwing check that will log the stack trace but not raise"""
    try:
        return check_addonpreferences(context.preferences)
    except RuntimeError:
        return None  # log stack trace but do not raise


def _get_media_duration(filepath: Path, found_media_files) -> int:
    import cv2

    def _cleanup(method: str):
        cap.release()

        logger.debug(
            f"Read video file '{filepath}' with the duration '{frame_count}' frames ({method})."
        )

        media = found_media_files.add()
        media.filepath = str(filepath)
        media.frame_count = frame_count

        return frame_count

    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open media file: {filepath}")

    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)  # seek to end
    duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps > 0 and duration_ms > 0:
        frame_count = int(round((duration_ms / 1000.0) * fps))
        if frame_count > 0:
            return _cleanup("fps + duration metadata")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if 0 < frame_count < 2**32 - 1:
        return _cleanup("frame count metadata")

    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.0)  # return seek to start

    frame_count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        frame_count += 1

    return _cleanup("manual")


def check_media_frame_counts(props: Mosplat_PG_Global, context: Context) -> None:
    def _on_failure(msg: str) -> NoReturn:
        props.do_media_durations_all_match = False
        props.computed_media_frame_count = -1
        raise RuntimeError(msg)

    props.found_media_files.clear()

    prefs = check_prefs_safe(context)
    if not prefs:
        _on_failure("Addon preferences not available in this context")

    media_dir_path = Path(props.current_media_dir)
    media_extension_set_str: str = prefs.media_extension_set

    extensions = [ext.strip().lower() for ext in media_extension_set_str.split(",")]

    files = [p for p in media_dir_path.iterdir() if p.suffix.lower() in extensions]

    if not files:
        _on_failure(
            f"No files found in the media directory with the preferred extensions: `{media_extension_set_str}`\n"
            f"Configure '{prefs.bl_rna.properties['media_extension_set'].name}' if needed."
        )

    logger.info(
        f"Reading frame counts of files within '{media_dir_path}'. This might take a while..."
    )

    frame_counts = [_get_media_duration(p, props.found_media_files) for p in files]

    if len(set(frame_counts)) != 1:
        _on_failure(
            f"Media files within '{media_dir_path}' should all have the same frame count."
        )

    props.do_media_durations_all_match = True
    props.computed_media_frame_count = frame_counts[0]
    logger.info(f"'{media_dir_path}' is a valid media directory.")
