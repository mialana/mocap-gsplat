# pyright: reportInvalidTypeForm=false
from __future__ import annotations
import bpy
from bpy.types import AddonPreferences
from bpy.props import (
    StringProperty,
)
from typing import Final
import os

ADDON_ID: Final[str] = __package__ or __name__


class Mosplat_AddonPreferences(AddonPreferences):
    bl_idname = ADDON_ID

    cache_dir: StringProperty(
        name="Cache Subdirectory",
        description="Cache directory used by the addon",
        default=bpy.utils.user_resource(
            "EXTENSIONS",
            path=os.path.join(".cache", ADDON_ID),
        ),
        subtype="DIR_PATH",
    )

    json_log_subdir: StringProperty(
        name="JSON Log Subdirectory",
        description="Subdirectory (relative to cache) for JSON logs",
        default="log",
        subtype="DIR_PATH",
    )

    json_log_filename_format: StringProperty(
        name="JSON Log Filename Format",
        description="strftime-compatible filename pattern",
        default="mosplat_%Y-%m-%d_%H-%M-%S.log",
    )

    json_date_log_format: StringProperty(
        name="JSON Date Format",
        description="strftime format for JSON log timestamps",
        default="%Y-%m-%d %H:%M:%S",
    )

    json_log_format: StringProperty(
        name="JSON Log Format",
        description="logging.Formatter format string",
        default="%(asctime)s %(levelname)s %(name)s %(filename)s %(funcName)s %(thread)d %(message)s",
    )

    stdout_date_log_format: StringProperty(
        name="STDOUT Date Format",
        description="strftime format for console log timestamps",
        default="%I:%M:%S %p",
    )

    stdout_log_format: StringProperty(
        name="STDOUT Log Format",
        description="logging.Formatter format string",
        default="[%(levelletter)s][%(asctime)s][%(dirname)s::%(filename)s::%(basename)s::%(funcName)s:%(lineno)s] %(message)s",
    )

    def draw(self, context):
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
