# pyright: reportInvalidTypeForm=false
from __future__ import annotations
from bpy.types import AddonPreferences, Context
from bpy.props import StringProperty, IntProperty

from pathlib import Path
import os
from typing import Set, Optional

from ..interfaces.logging_interface import MosplatLoggingInterface
from ..infrastructure.mixins import (
    MosplatBlPropertyAccessorMixin,
    MosplatPGAccessorMixin,
)
from ..infrastructure.constants import (
    ADDON_PREFERENCES_ID,
    ADDON_BASE_ID,
    DEFAULT_PREPROCESS_MEDIA_SCRIPT_FILE,
    ADDON_SHORTNAME,
)
from ..infrastructure.schemas import UserFacingError


def update_stdout_logging(prefs: Mosplat_AP_Global, _: Context):
    if MosplatLoggingInterface.init_stdout_handler(
        log_fmt=prefs.stdout_log_format,
        log_date_fmt=prefs.stdout_date_log_format,
    ):
        prefs.logger().info("STDOUT logging updated.")


def update_json_logging(prefs: Mosplat_AP_Global, _: Context):
    if MosplatLoggingInterface.init_json_handler(
        log_fmt=prefs.json_log_format,
        log_date_fmt=prefs.json_date_log_format,
        outdir=prefs.json_log_dir,
        file_fmt=prefs.json_log_filename_format,
    ):
        prefs.logger().info("JSON logging updated.")


class Mosplat_AP_Global(
    AddonPreferences, MosplatBlPropertyAccessorMixin, MosplatPGAccessorMixin
):
    bl_idname = ADDON_PREFERENCES_ID

    cache_dir: StringProperty(
        name="Cache Directory",
        description="Cache directory on disk used by the addon",
        default=str(Path.home().joinpath(".cache", ADDON_BASE_ID)),
        subtype="DIR_PATH",
        update=update_json_logging,
    )

    json_log_subdir: StringProperty(
        name="JSON Log Subdirectory",
        description="Subdirectory (relative to cache) for JSON logs",
        default="log",
        update=update_json_logging,
    )

    vggt_model_subdir: StringProperty(
        name="Model Cache Subdirectory",
        description="Subdirectory where the VGGT model data will be stored",
        default="vggt",
    )

    data_output_path: StringProperty(
        name="Data Output Path",
        description="Output directory for processed data generated from the selected media directory.\n"
        "Relative paths are resolved against the selected media directory.\n"
        "The token {media_directory_name} will be replaced with the base name of the selected media directory.",
        default=f"{os.curdir}{os.sep}{{media_directory_name}}_OUTPUT",
    )

    preprocess_media_script_file: StringProperty(
        name="Preprocess Media Script File",
        description=f"A file containing a Python script that will be applied to the contents of the selected media directory before individual frames are extracted.\n"
        "See '{DEFAULT_PREPROCESS_MEDIA_SCRIPT_FILE}' for details on the expected format of the file.\n"
        "If an empty path is entered no pre-processing will be performed.",
        default=DEFAULT_PREPROCESS_MEDIA_SCRIPT_FILE,
        subtype="FILE_PATH",
    )

    media_extensions: StringProperty(
        name="Media Extensions",
        description="Comma separated string of all file extensions that should be considered as media files within the media directory",
        default=".avi,.mp4,.mov",
    )

    max_frame_range: IntProperty(
        name="Max Frame Range",
        description="The max frame range that can be processed at once.\n"
        "Set to '-1' for no limit.\n"
        "WARNING: Change this preference with knowledge of the capabilities of your own machine.",
        default=120,
    )

    json_log_filename_format: StringProperty(
        name="JSON Log Filename Format",
        description="strftime-compatible filename pattern",
        default="mosplat_%Y-%m-%d_%H-%M-%S.log",
        update=update_json_logging,
    )

    json_date_log_format: StringProperty(
        name="JSON Date Format",
        description="strftime format for JSON log timestamps",
        default="%Y-%m-%d %H:%M:%S",
        update=update_json_logging,
    )

    json_log_format: StringProperty(
        name="JSON Log Format",
        description=f"`logging.Formatter` format string. Refer to `{MosplatLoggingInterface.set_log_record_factory.__qualname__}` for info about custom logrecord attributes: `levelletter`, `dirname`, and `classname`.",
        default="%(asctime)s %(levelname)s %(name)s %(pathname)s %(classname)s %(funcName)s %(lineno)s %(thread)d %(message)s",
        update=update_json_logging,
    )

    stdout_date_log_format: StringProperty(
        name="STDOUT Date Format",
        description="strftime format for console log timestamps",
        default="%I:%M:%S %p",
        update=update_stdout_logging,
    )

    stdout_log_format: StringProperty(
        name="STDOUT Log Format",
        description=f"`logging.Formatter` format string. Refer to `{MosplatLoggingInterface.set_log_record_factory.__qualname__}` for info about custom logrecord attributes: `levelletter`, `dirname`, and `classname`.",
        default="[%(levelletter)s][%(asctime)s][%(dirname)s::%(filename)s::%(classname)s::%(funcName)s:%(lineno)s] %(message)s",
        update=update_stdout_logging,
    )

    vggt_hf_id: StringProperty(
        name="VGGT Hugging Face ID",
        description="ID of VGGT pre-trained model on Hugging Face",
        default="facebook/VGGT-1B",
    )

    @property
    def json_log_dir(self) -> Path:
        return Path(self.cache_dir).joinpath(self.json_log_subdir)

    @property
    def vggt_model_dir(self) -> Path:
        return Path(self.cache_dir).joinpath(self.vggt_model_subdir)

    @property
    def media_extensions_set(self) -> Set[str]:
        try:
            return set(
                [ext.strip().lower() for ext in str(self.media_extensions).split(",")]
            )
        except IndexError:
            raise UserFacingError(
                f"Extensions in '{self.get_prop_name('media_extensions')}' should be separated by commas."
            )

    def draw(self, _: Context):
        layout = self.layout

        layout.label(
            text=f"{ADDON_SHORTNAME.capitalize()} Saved Preferences", icon="SETTINGS"
        )

        col = layout.column(align=True)

        gen_io_box = col.box()
        gen_io_box.label(text="General I/O Configuration", icon="DISK_DRIVE_LARGE")
        gen_io_box.prop(self, "cache_dir")
        gen_io_box.prop(self, "json_log_subdir")
        gen_io_box.prop(self, "vggt_model_subdir")

        layout.separator()

        data_proc_box = col.box()
        data_proc_box.label(text="Data Processing Configuration", icon="MESH_CYLINDER")
        data_proc_box.prop(self, "data_output_path")
        data_proc_box.prop(self, "preprocess_media_script_file")
        data_proc_box.prop(self, "media_extensions")
        data_proc_box.prop(self, "max_frame_range")

        layout.separator()

        json_box = col.box()
        json_box.label(text="JSON Log Configuration", icon="FILE_TEXT")
        json_box.prop(self, "json_log_filename_format")
        json_box.prop(self, "json_date_log_format")
        json_box.prop(self, "json_log_format")

        layout.separator()

        stdout_box = col.box()
        stdout_box.label(text="STDOUT Log Formatting", icon="GREASEPENCIL")
        stdout_box.prop(self, "stdout_date_log_format")
        stdout_box.prop(self, "stdout_log_format")
