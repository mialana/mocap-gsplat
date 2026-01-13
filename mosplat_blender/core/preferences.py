# pyright: reportInvalidTypeForm=false
from __future__ import annotations
from bpy.types import AddonPreferences, Context
from bpy.props import (
    StringProperty,
)

from pathlib import Path

from ..interfaces.logging_interface import MosplatLoggingInterface
from ..infrastructure.mixins import MosplatLogClassMixin
from ..infrastructure.constants import ADDON_ID


def update_stdout_logging(prefs: Mosplat_AP_Global, _: Context):
    if MosplatLoggingInterface.init_stdout_handler(
        log_fmt=prefs.stdout_log_format,
        log_date_fmt=prefs.stdout_date_log_format,
    ):
        prefs.logger().info("STDOUT logging updated.")
    else:
        raise TypeError()


def update_json_logging(prefs: Mosplat_AP_Global, _: Context):
    outdir: Path = Path(prefs.cache_dir).joinpath(prefs.json_log_subdir)
    if MosplatLoggingInterface.init_json_handler(
        log_fmt=prefs.json_log_format,
        log_date_fmt=prefs.json_date_log_format,
        outdir=outdir,
        file_fmt=prefs.json_log_filename_format,
    ):
        prefs.logger().info("JSON logging updated.")


class Mosplat_AP_Global(AddonPreferences, MosplatLogClassMixin):
    bl_idname = ADDON_ID

    cache_dir: StringProperty(
        name="Cache Directory",
        description="Cache directory used by the addon",
        default=str(Path.home().joinpath(".cache", ADDON_ID)),
        subtype="DIR_PATH",
        update=update_json_logging,
    )

    json_log_subdir: StringProperty(
        name="JSON Log Subdirectory",
        description="Subdirectory (relative to cache) for JSON logs",
        default="log",
        update=update_json_logging,
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

    def draw(self, _: Context):
        layout = self.layout

        layout.label(text="Output Configuration", icon="FILE_FOLDER")

        col = layout.column(align=True)
        col.prop(self, "cache_dir")
        col.prop(self, "json_log_subdir")

        layout.separator()

        col = layout.column(align=True)
        col.label(text="JSON Log Configuration", icon="TEXT")
        col.prop(self, "json_log_filename_format")
        col.prop(self, "json_date_log_format")
        col.prop(self, "json_log_format")

        layout.separator()

        col = layout.column(align=True)
        col.label(text="STDOUT Log Formatting", icon="TEXT")
        col.prop(self, "stdout_date_log_format")
        col.prop(self, "stdout_log_format")
