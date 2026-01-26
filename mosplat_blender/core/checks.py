"""
functions here perform runtime type-checking and raise appropriate errors
if items in the Blender ecosystem are not as expected.

if they do successfuly pass though, they perform static casting so that
we can operate with type-awareness in development.
"""

from __future__ import annotations

from bpy.types import Preferences, Scene, Context

import os
from pathlib import Path
from typing import cast, TYPE_CHECKING, Optional, Set, List

from ..infrastructure.constants import (
    ADDON_PREFERENCES_ID,
    ADDON_PROPERTIES_ATTRIBNAME,
    MEDIA_IO_DATASET_JSON_FILENAME,
)
from ..infrastructure.schemas import UnexpectedError, SafeError, UserFacingError
from ..interfaces import MosplatLoggingInterface

if TYPE_CHECKING:
    from .preferences import Mosplat_AP_Global
    from .properties import Mosplat_PG_Global

logger = MosplatLoggingInterface.configure_logger_instance(__name__)


def check_propertygroup(
    scene: Optional[Scene],
) -> Mosplat_PG_Global:
    if scene is None:  # can occur if checked at the wrong time
        raise SafeError("Blender scene unavailable in this context.")
    try:
        found_properties = getattr(scene, ADDON_PROPERTIES_ATTRIBNAME)

        if TYPE_CHECKING:
            # OK to use `cast` here as we've guarded its existence with a try-block, and we created it
            found_properties = cast(Mosplat_PG_Global, found_properties)

        return found_properties
    except AttributeError as e:
        raise UnexpectedError(
            "Registration of addon properties was never successful. Cannot continue."
        ) from e


def check_addonpreferences(
    prefs_ctx: Optional[Preferences],
) -> Mosplat_AP_Global:
    if prefs_ctx is None:  # can occur if checked at the wrong time
        raise SafeError("Blender preferences unavailable in this context.")

    try:
        found_preferences = prefs_ctx.addons[ADDON_PREFERENCES_ID].preferences

        if TYPE_CHECKING:
            # OK to use `cast` here as we've guarded its existence with a try-block, and we created it
            found_preferences = cast(Mosplat_AP_Global, found_preferences)
        return found_preferences
    except KeyError as e:
        raise UnexpectedError(
            "Registration of addon preferences was never successful. Cannot continue."
        ) from e


def check_props_safe(context: Context) -> Optional[Mosplat_PG_Global]:
    """provide a safe, non-throwing check that will log the stack trace but not raise"""
    try:
        return check_propertygroup(context.scene)
    except SafeError as e:
        logger.warning(str(e))
        return None  # log stack trace but do not raise


def check_prefs_safe(context: Context) -> Optional[Mosplat_AP_Global]:
    """provide a safe, non-throwing check that will log the stack trace but not raise"""
    try:
        return check_addonpreferences(context.preferences)
    except SafeError as e:
        logger.warning(str(e))
        return None  # log stack trace but do not raise


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
        output = current_media_dirpath.joinpath(formatted_output_path)

    try:
        os.makedirs(
            output, exist_ok=True
        )  # see if the directory can be created successfully
    except (FileExistsError, PermissionError, OSError):
        raise UserFacingError(
            f"'{props.get_prop_name('current_media_dir')}' and '{prefs.get_prop_name('data_output_path')}' create an invalid directory value: '{output}'"
        )

    return output


def check_data_json_filepath(prefs: Mosplat_AP_Global, props: Mosplat_PG_Global):
    return check_data_output_dirpath(prefs, props).joinpath(
        MEDIA_IO_DATASET_JSON_FILENAME
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
    try:
        exts = set(
            [ext.strip().lower() for ext in str(prefs.media_extensions).split(",")]
        )
    except IndexError:
        raise UserFacingError(
            f"Extensions in '{prefs.get_prop_name('media_extensions')}' should be separated by commas."
        )
    if len(exts) == 0:
        raise UserFacingError(
            f"No extensions could be parsed from '{prefs.get_prop_name('media_extensions')}'."
        )
    return exts


def check_current_media_dirpath(props: Mosplat_PG_Global):
    dirpath = Path(props.current_media_dir)
    if not dirpath.is_dir():
        raise UserFacingError(
            f"'{props.get_prop_name('media_dir_path')}' is not a valid directory."
        )

    return dirpath
