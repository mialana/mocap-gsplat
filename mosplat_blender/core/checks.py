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

from infrastructure.constants import PER_FRAME_DIRNAME
from infrastructure.macros import try_access_path
from infrastructure.schemas import (
    AddonMeta,
    NPZNameToPathLookup,
    SavedNPZName,
    UnexpectedError,
    UserFacingError,
)
from interfaces import MosplatLoggingInterface

if TYPE_CHECKING:
    from bpy.types import Preferences, Scene, WindowManager

    from core.preferences import Mosplat_AP_Global
    from core.properties import Mosplat_PG_Global

logger = MosplatLoggingInterface.configure_logger_instance(__name__)

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
        found_preferences = prefs_ctx.addons[_ADDON_META.global_prefs_id].preferences

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


def check_data_output_dirpath(
    prefs: Mosplat_AP_Global, props: Mosplat_PG_Global
) -> Path:
    output: Path

    current_media_dirpath = check_current_media_dirpath(props)
    media_directory_name = current_media_dirpath.name

    formatted_output_path = Path(
        str(prefs.data_output_path).format(media_directory_name=media_directory_name)
    )
    if formatted_output_path.is_absolute():
        output = formatted_output_path
    else:
        output = current_media_dirpath / formatted_output_path

    try:
        os.makedirs(
            output, exist_ok=True
        )  # see if the directory can be created successfully
    except (FileExistsError, PermissionError, OSError):
        raise UserFacingError(
            f"'{props._meta.current_media_dir.name}' and '{prefs._meta.data_output_path.name}' create an invalid directory value: '{output}'"
        )

    return output


def check_data_json_filepath(prefs: Mosplat_AP_Global, props: Mosplat_PG_Global):
    return (
        check_data_output_dirpath(prefs, props) / _ADDON_META.media_io_dataset_filename
    )


def check_media_files(prefs: Mosplat_AP_Global, props: Mosplat_PG_Global) -> List[Path]:
    exts = check_media_extensions_set(prefs)

    files = sorted(
        [
            p
            for p in check_current_media_dirpath(props).iterdir()
            if p.suffix.lower() in exts
        ]
    )

    if len(files) == 0:
        raise UserFacingError(
            f"No files were found in '{props.current_media_dir}' with extensions of '{prefs.media_extensions}'."
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


def check_current_media_dirpath(props: Mosplat_PG_Global):
    dirpath = Path(props.current_media_dir)
    if not dirpath.is_dir():
        raise UserFacingError(
            f"'{props._meta.current_media_dir.name}' is not a valid directory."
        )

    return dirpath


def check_frame_range_poll_result(
    prefs: Mosplat_AP_Global, props: Mosplat_PG_Global
) -> List[str]:
    err_list = []
    start, end = props.current_frame_range
    curr_range_name = props._meta.current_frame_range.name
    if start >= end:
        err_list.append(
            f"Start frame for '{curr_range_name}' must be less than end frame."
        )

    if end >= props.dataset_accessor.median_frame_count:
        prop_name = props.dataset_accessor._meta.median_frame_count.name
        count = props.dataset_accessor.median_frame_count
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


def check_npz_filepaths_for_frame_range(
    prefs: Mosplat_AP_Global,
    props: Mosplat_PG_Global,
    names: List[SavedNPZName],
    should_exist: List[bool],
) -> NPZNameToPathLookup:
    """
    generates NPZ file paths given a list of desired names and for all frames in current
    frame range. Use `should_exist` if the npz files should already exist to raise
    `UserFacingError` if an NPZ file does not exist where it is expected to.
    """
    start, end = props.current_frame_range

    data_dir = check_data_output_dirpath(prefs, props)

    all_npz_files: NPZNameToPathLookup = {name: [] for name in names}

    for frame in range(start, end):
        frame_dir = data_dir / PER_FRAME_DIRNAME.format(frame)

        for idx, name in enumerate(names):
            npz_file = frame_dir / f"{name}.npz"
            if should_exist[idx]:
                try:
                    try_access_path(npz_file)  # raises the below exceptions
                except (OSError, PermissionError, FileNotFoundError) as e:
                    raise UserFacingError(
                        f"Expected to find an extracted NPZ file at '{npz_file}'."
                        "Re-extract the frame range if necessary.",
                        e,
                    )  # convert now to a `UserFacingError`
            all_npz_files[name].append(npz_file)

    return all_npz_files
