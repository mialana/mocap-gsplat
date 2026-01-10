import bpy
import sys
import logging
from pythonjsonlogger.json import JsonFormatter
from typing import TYPE_CHECKING, TypeAlias, Any

from .base_ot import Mosplat_OT_Base
from ..constants import LOGGER_NAME, DATE_LOG_FORMAT, JSON_LOG_FORMAT, STDOUT_LOG_FORMAT
from ..properties import MosplatProperties

if TYPE_CHECKING:
    from bpy.stub_internal.rna_enums import OperatorReturnItems
else:
    OperatorReturnItems: TypeAlias = Any


class OT_InitLogging(Mosplat_OT_Base):
    short_name = "logging"

    @classmethod
    def poll(cls, context: bpy.types.Context):
        # if scene does not exist or properties have not been registered this op is not valid
        return bool(context.scene) | isinstance(
            getattr(context.scene, "mosplat_properties", None), MosplatProperties
        )

    def execute(self, context: bpy.types.Context):
        return self._init_logging(context.scene)

    def invoke(
        self, context: bpy.types.Context, event: bpy.types.Event
    ) -> set[OperatorReturnItems]:
        return self._init_logging(context.scene)

    def _init_logging(self, scene: bpy.types.Scene | None) -> set[OperatorReturnItems]:
        props: MosplatProperties = getattr(scene, "mosplat_properties")
        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(logging.DEBUG)

        stdout_formatter = logging.Formatter(STDOUT_LOG_FORMAT, datefmt=DATE_LOG_FORMAT)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(stdout_formatter)
        logger.addHandler(stdout_handler)

        json_formatter = JsonFormatter(JSON_LOG_FORMAT, datefmt=DATE_LOG_FORMAT)
        json_handler = logging.FileHandler(props.logging_output)
        json_handler.setFormatter(json_formatter)

        logger.addHandler(json_handler)

        # store handlers in driver namespace
        dns = bpy.app.driver_namespace
        dns["mosplat_stdout_log_handler"] = stdout_handler
        dns["mosplat_json_log_handler"] = json_handler

        logging.info("Mosplat custom logging initiated")

        return {"FINISHED"}


def deinit_logging():
    logger = logging.getLogger(LOGGER_NAME)

    # remove logging handlers
    dns = bpy.app.driver_namespace
    if (
        hasattr(dns, "mosplat_stdout_log_handler")
        and dns["mosplat_stdout_log_handler"] is not None
    ):
        logger.removeHandler(dns["mosplat_stdout_log_handler"])
    if (
        hasattr(dns, "mosplat_json_log_handler")
        and dns["mosplat_json_log_handler"] is not None
    ):
        logger.removeHandler(dns["mosplat_json_log_handler"])
