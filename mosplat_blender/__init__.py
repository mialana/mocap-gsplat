import bpy
import sys
import logging
from pythonjsonlogger.json import JsonFormatter

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
json_formatter = JsonFormatter(LOG_FORMAT)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(json_formatter)
logger.addHandler(stdout_handler)

def register():
    logger.info("MOSPLAT Blender addon registration starting.")


def unregister():
    logger.info("MOSPLAT Blender addon unregistration completed.")
