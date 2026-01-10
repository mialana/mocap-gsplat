import bpy

import os
import sys
import logging
import datetime
from typing import Tuple
from pythonjsonlogger.json import JsonFormatter
import colorlog

from .constants import (
    LOGGER_NAME,
    STDOUT_DATE_LOG_FORMAT,
    STDOUT_LOG_FORMAT,
    JSON_DATE_LOG_FORMAT,
    JSON_LOG_FORMAT,
    JSON_LOG_OUTSUBDIR,
    JSON_LOG_OUTFILE,
)


class MosplatFormatter(colorlog.ColoredFormatter):
    def format(self, record: logging.LogRecord) -> str:
        # 1-letter log level
        record.levelname = record.levelname[0]

        # truncate path to directory + filename
        record.shortpath = os.path.join(
            os.path.basename(os.path.dirname(record.pathname)),
            os.path.basename(record.pathname),
        )

        return super().format(record)


def init_logging() -> Tuple[logging.Logger, logging.StreamHandler, logging.FileHandler]:

    logger: logging.Logger = logging.getLogger(LOGGER_NAME)
    # set logger to most verbose
    logger.setLevel(logging.DEBUG)

    # set formatter of stdout logger and add as handler
    stdout_log_formatter = MosplatFormatter(
        STDOUT_LOG_FORMAT, datefmt=STDOUT_DATE_LOG_FORMAT
    )
    stdout_log_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    stdout_log_handler.setFormatter(stdout_log_formatter)
    logger.addHandler(stdout_log_handler)

    # build path for json log output file, set formatter, and add as handler
    json_log_formatter = JsonFormatter(JSON_LOG_FORMAT, datefmt=JSON_DATE_LOG_FORMAT)
    json_log_outdir = bpy.utils.user_resource(
        "EXTENSIONS",
        path=str(JSON_LOG_OUTSUBDIR),
    )  # full path to log directory
    os.makedirs(json_log_outdir, exist_ok=True)  # make the log directory if necessary
    json_log_outfile = os.path.join(
        json_log_outdir, datetime.datetime.now().strftime(JSON_LOG_OUTFILE)
    )

    json_log_handler: logging.FileHandler = logging.FileHandler(json_log_outfile)
    json_log_handler.setFormatter(json_log_formatter)
    logger.addHandler(json_log_handler)

    logger.info("Custom Mosplat logging instantiated.")

    return (logger, stdout_log_handler, json_log_handler)
