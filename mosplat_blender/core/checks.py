"""
functions here perform runtime type-checking and raise appropriate errors
if items in the Blender ecosystem are not as expected.

if they do successfuly pass though, they perform static casting so that
we can operate with type-awareness in development.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Set, TypeGuard, cast

from ..infrastructure.schemas import (
    AddonMeta,
    UnexpectedError,
    UserFacingError,
)
from ..interfaces.logging_interface import LoggingInterface

if TYPE_CHECKING:
    from bpy.types import Preferences, Scene, WindowManager

    from .preferences import Mosplat_AP_Global
    from .properties import Mosplat_PG_Global

logger = LoggingInterface.configure_logger_instance(__name__)

_ADDON_META = AddonMeta()


def check_propertygroup(
    scene: Optional[Scene],
) -> Mosplat_PG_Global:
    if scene is None:  # can occur if checked at the wrong time
        raise UserFacingError("Blender scene unavailable in this context.")
    try:
        found_properties = getattr(scene, _ADDON_META.global_props_name)

        if TYPE_CHECKING:
            # OK to use `cast` here as we've guarded its existence with a try-block, and we created it
            found_properties = cast(Mosplat_PG_Global, found_properties)

        return found_properties
    except AttributeError as e:
        raise UnexpectedError(
            "Registration of addon properties was never successful. Cannot continue.", e
        ) from e


def check_addonpreferences(
    prefs_ctx: Optional[Preferences],
) -> Mosplat_AP_Global:
    if prefs_ctx is None:  # can occur if checked at the wrong time
        raise UserFacingError("Blender preferences unavailable in this context.")

    try:
        id: str = _ADDON_META.global_runtime_module_id
        found_preferences = prefs_ctx.addons[id].preferences

        if TYPE_CHECKING:
            # OK to use `cast` here as we've guarded its existence with a try-block, and we created it
            found_preferences = cast(Mosplat_AP_Global, found_preferences)
        return found_preferences
    except KeyError as e:
        raise UnexpectedError(
            "Registration of addon prefs was never successful. Cannot continue.", e
        ) from e


def check_window_manager(wm: Optional[WindowManager]) -> TypeGuard[WindowManager]:
    if wm is None:
        raise UserFacingError("Window manager unavailable in this context.")
    return True


def check_media_directory(props: Mosplat_PG_Global):
    dirpath = Path(props.media_directory)
    if not dirpath.is_dir():
        raise UserFacingError(
            f"'{props._meta.media_directory.name}' is not a valid directory."
        )

    return dirpath


def check_media_output_dir(prefs: Mosplat_AP_Global, props: Mosplat_PG_Global) -> Path:
    output: Path

    media_directory = check_media_directory(props)
    media_directory_name = media_directory.name

    formatted_output_path = Path(
        str(prefs.media_output_dir_format).format(
            media_directory_name=media_directory_name
        )
    )
    if formatted_output_path.is_absolute():
        output = formatted_output_path
    else:
        output = media_directory / formatted_output_path

    try:
        os.makedirs(
            output, exist_ok=True
        )  # see if the directory can be created successfully
    except (FileExistsError, PermissionError, OSError):
        raise UserFacingError(
            f"'{props._meta.media_directory.name}' and '{prefs._meta.media_output_dir_format.name}' create an invalid directory value: '{output}'"
        )

    return output


def check_media_io_metadata_filepath(
    prefs: Mosplat_AP_Global, props: Mosplat_PG_Global
):
    return check_media_output_dir(prefs, props) / _ADDON_META.media_io_metadata_filename


def check_media_files(prefs: Mosplat_AP_Global, props: Mosplat_PG_Global) -> List[Path]:
    exts = check_media_extensions_set(prefs)

    files = sorted(
        [p for p in check_media_directory(props).iterdir() if p.suffix.lower() in exts]
    )

    if len(files) == 0:
        raise UserFacingError(
            f"No files were found in '{props.media_directory}' with extensions of '{prefs.media_extensions}'."
        )
    return files


def check_media_extensions_set(prefs: Mosplat_AP_Global) -> Set[str]:
    prop_name = prefs._meta.media_extensions.name
    try:
        exts = set(
            [ext.strip().lower() for ext in str(prefs.media_extensions).split(",")]
        )
    except IndexError:
        raise UserFacingError(
            f"Extensions in '{prop_name}' should be separated by commas."
        )
    if len(exts) == 0:
        raise UserFacingError(f"No extensions could be parsed from '{prop_name}'.")
    return exts


def check_frame_range_poll_result(
    prefs: Mosplat_AP_Global, props: Mosplat_PG_Global
) -> List[str]:
    err_list = []
    start, end = props.frame_range_
    curr_range_name = props._meta.media_directory.name
    if start >= end:
        err_list.append(
            f"Start frame for '{curr_range_name}' must be less than end frame."
        )

    if end >= props.metadata_accessor.median_frame_count:
        prop_name = props.metadata_accessor._meta.median_frame_count.name
        count = props.metadata_accessor.median_frame_count
        err_list.append(
            f"End frame must be less than '{prop_name}' of '{count}' frames."
        )

    max_frame_range = prefs.max_frame_range
    max_range_name = prefs._meta.max_frame_range
    if max_frame_range != -1 and end - start > prefs.max_frame_range:
        err_list.append(
            f"For best results, set '{curr_range_name}' to less than '{max_frame_range}'.\n"
            f"Customize this restriction in the addon's preferences under '{max_range_name}'"
        )

    return err_list  # return even if list if empty
