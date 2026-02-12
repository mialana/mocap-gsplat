# pyright: reportInvalidTypeForm=false
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Set

from bpy.props import IntProperty, StringProperty
from bpy.types import AddonPreferences, Context

from mosplat_blender.core.checks import check_media_extensions_set
from mosplat_blender.core.meta.preferences_meta import (
    MOSPLAT_AP_GLOBAL_META,
    Mosplat_AP_Global_Meta,
)
from mosplat_blender.infrastructure.constants import (
    DEFAULT_MAX_LOG_ENTRIES,
    DEFAULT_PREPROCESS_SCRIPT,
    PENN_DEFAULT_PREPROCESS_SCRIPT,
)
from mosplat_blender.infrastructure.identifiers import OperatorIDEnum
from mosplat_blender.infrastructure.macros import try_access_path
from mosplat_blender.infrastructure.mixins import EnforceAttributesMixin
from mosplat_blender.infrastructure.schemas import (
    AddonMeta,
    UnexpectedError,
    UserFacingError,
)
from mosplat_blender.interfaces import LoggingInterface, VGGTInterface

if TYPE_CHECKING:
    from mosplat_blender.core.preferences import Mosplat_AP_Global

LoggingHandler = LoggingInterface.LoggingHandler


def make_update_logging_fn(handler: LoggingHandler):
    def update_logging(self: Mosplat_AP_Global, _: Context):
        try:
            LoggingInterface().init_handler(handler, self)
            self.logger.info(f"{handler} logging updated.")
        except UserFacingError as e:
            self.logger.error(
                UserFacingError.make_msg(f"{handler} log settings invalid.", e)
            )

    return update_logging


def update_media_extensions(self: Mosplat_AP_Global, context: Context):
    OperatorIDEnum.run(OperatorIDEnum.VALIDATE_FILE_STATUSES)
    self.logger.info(f"'{self._meta.media_extensions.name}' updated.")


def update_model_preferences(self: Mosplat_AP_Global, context: Context):
    if VGGTInterface().model is not None:
        if VGGTInterface().model_cache_dir != self.cache_dir_vggt_:
            self.logger.warning(
                "Changed model cache dir, next init will take a while."
                "Change back to previous directory if this is not desired."
            )
        elif VGGTInterface().hf_id != self.vggt_hf_id:
            self.logger.warning(
                "Changed Hugging Face ID for model, next init will take a while."
                "Change back to previous ID if this is not desired."
            )
        else:
            return  # prefs did not change from what is currently initialized with.
        try:
            VGGTInterface.cleanup_interface()
        except UnexpectedError as e:
            self.logger.warning(str(e))


class Mosplat_AP_Global(AddonPreferences, EnforceAttributesMixin):
    _meta: Mosplat_AP_Global_Meta = MOSPLAT_AP_GLOBAL_META
    bl_idname = AddonMeta().global_prefs_id

    cache_dir: StringProperty(
        name="Cache Directory",
        description="Cache directory on disk used by the addon",
        default=str(Path.home() / ".cache" / AddonMeta().base_id),
        subtype="DIR_PATH",
        update=make_update_logging_fn(handler=LoggingHandler.JSON),
    )

    cache_subdir_logs: StringProperty(
        name="Logs Cache Subdirectory",
        description="Subdirectory in cache directory for JSON logs",
        default="log",
        update=make_update_logging_fn(handler=LoggingHandler.JSON),
    )

    cache_subdir_model: StringProperty(
        name="Model Cache Subdirectory",
        description="Subdirectory in cache directory where model data will be stored",
        default="vggt",
        update=update_model_preferences,
    )

    media_output_dir_format: StringProperty(
        name="Media Output Directory Format",
        description="Format for resolving the path to a directory where processed data generated for the selected media directory will be outputted.\n"
        "Relative paths are resolved against the selected media directory path.\n"
        "The token {media_directory_name} will be replaced with the base name of the selected media directory.",
        default=f"{os.curdir}{os.sep}{{media_directory_name}}_OUTPUT",
    )

    preprocess_media_script_file: StringProperty(
        name="Preprocess Media Script File",
        description=f"A file containing a Python script that will be applied to the contents of the selected media directory before individual frames are extracted.\n"
        f"See '{DEFAULT_PREPROCESS_SCRIPT}' for details on the expected format of the file.\n"
        "If an empty path is entered no pre-processing will be performed.",
        default=PENN_DEFAULT_PREPROCESS_SCRIPT,
        subtype="FILE_PATH",
    )

    media_extensions: StringProperty(
        name="Media Extensions",
        description="Comma separated string of all file extensions that should be considered as media files within the media directory",
        default=".avi,.mp4,.mov",
        update=update_media_extensions,
    )

    max_frame_range: IntProperty(
        name="Max Frame Range",
        description="The max frame range that can be processed at once.\n"
        "Set to '-1' for no limit.\n"
        "WARNING: Change this preference with knowledge of the capabilities of your own machine.",
        default=120,
    )

    json_log_filename_format: StringProperty(
        name="JSON Logging Filename Format",
        description="strftime-compatible filename pattern",
        default="mosplat_%Y-%m-%d_%H-%M-%S.log",
        update=make_update_logging_fn(handler=LoggingHandler.JSON),
    )

    json_date_log_format: StringProperty(
        name="JSON Logging Date Format",
        description="strftime format for JSON log timestamps",
        default="%Y-%m-%d %H:%M:%S",
        update=make_update_logging_fn(handler=LoggingHandler.JSON),
    )

    json_log_format: StringProperty(
        name="JSON Logging Format",
        description=f"`logging.Formatter` format string. Refer to `{LoggingInterface._set_log_record_factory.__qualname__}` for info about custom logrecord attributes: `levelletter`, `dirname`, and `classname`.",
        default="%(asctime)s %(levelname)s %(name)s %(pathname)s %(classname)s %(funcName)s %(lineno)s %(thread)d %(message)s",
        update=make_update_logging_fn(handler=LoggingHandler.JSON),
    )

    stdout_date_log_format: StringProperty(
        name="STDOUT Logging Date Format",
        description="strftime format for console log timestamps",
        default="%I:%M:%S %p",
        update=make_update_logging_fn(handler=LoggingHandler.STDOUT),
    )

    stdout_log_format: StringProperty(
        name="STDOUT Logging Log Format",
        description=f"`logging.Formatter` format string. Refer to `{LoggingInterface._set_log_record_factory.__qualname__}` for info about custom logrecord attributes: `levelletter`, `dirname`, and `classname`.",
        default="[%(levelletter)s][%(asctime)s][%(dirname)s::%(filename)s::%(classname)s::%(funcName)s:%(lineno)s] %(message)s",
        update=make_update_logging_fn(handler=LoggingHandler.STDOUT),
    )

    blender_max_log_entries: IntProperty(
        name="BLENDER Logging Max Entries",
        description="The max amount of log entries that will be stored as Blender data and displayed in UI.",
        default=DEFAULT_MAX_LOG_ENTRIES,
        update=make_update_logging_fn(handler=LoggingHandler.BLENDER),
    )

    vggt_hf_id: StringProperty(
        name="VGGT Hugging Face ID",
        description="ID of VGGT pre-trained model on Hugging Face",
        default="facebook/VGGT-1B",
        update=update_model_preferences,
    )

    @property
    def cache_dir_logs_(self) -> Path:
        return Path(self.cache_dir) / self.cache_subdir_logs

    @property
    def cache_dir_vggt_(self) -> Path:
        return Path(self.cache_dir) / self.cache_subdir_model

    @property
    def preprocess_media_script_file_(self) -> Path:
        """raises `UserFacingError` if the script is not accessible."""
        file_path = Path(self.preprocess_media_script_file)

        try:
            try_access_path(file_path)
            return file_path
        except (OSError, PermissionError, FileNotFoundError) as e:
            raise UserFacingError(
                f"Given preprocess script file is not accessible: '{file_path}'", e
            ) from e  # convert now to a `UserFacingError`

    @property
    def media_extensions_set(self) -> Set[str]:
        return check_media_extensions_set(self)

    def draw(self, _: Context):
        layout = self.layout

        layout.label(
            text=f"{AddonMeta().shortname.capitalize()} Saved Preferences",
            icon="SETTINGS",
        )

        col = layout.column(align=True)

        meta = self._meta

        gen_io_box = col.box()
        gen_io_box.label(text="General I/O Configuration", icon="DISK_DRIVE_LARGE")
        gen_io_box.prop(self, meta.cache_dir.id)
        gen_io_box.prop(self, meta.cache_subdir_logs.id)
        gen_io_box.prop(self, meta.cache_subdir_model.id)
        gen_io_box.prop(self, meta.vggt_hf_id.id)

        layout.separator()

        data_proc_box = col.box()
        data_proc_box.label(text="Data Processing Configuration", icon="MESH_CYLINDER")
        data_proc_box.prop(self, meta.media_output_dir_format.id)
        data_proc_box.prop(self, meta.preprocess_media_script_file.id)
        data_proc_box.prop(self, meta.media_extensions.id)
        data_proc_box.prop(self, meta.max_frame_range.id)

        layout.separator()

        json_box = col.box()
        json_box.label(text="JSON Logging Configuration", icon="FILE_TEXT")
        json_box.prop(self, meta.json_log_filename_format.id)
        json_box.prop(self, meta.json_date_log_format.id)
        json_box.prop(self, meta.json_log_format.id)

        layout.separator()

        stdout_box = col.box()
        stdout_box.label(text="STDOUT Logging Configuration", icon="GREASEPENCIL")
        stdout_box.prop(self, meta.stdout_date_log_format.id)
        stdout_box.prop(self, meta.stdout_log_format.id)

        layout.separator()

        blender_box = col.box()
        blender_box.label(text="BLENDER Logging Configuration", icon="GREASEPENCIL")
        blender_box.prop(self, meta.blender_max_log_entries.id)
