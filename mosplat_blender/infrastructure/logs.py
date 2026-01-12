import bpy

import os
import sys
import logging
import datetime
from pathlib import Path
from typing import ClassVar
from pythonjsonlogger.json import JsonFormatter
import coloredlogs
from humanfriendly.compat import coerce_string
from humanfriendly.terminal import ansi_wrap

from dataclasses import dataclass

from .constants import (
    STDOUT_DATE_LOG_FORMAT,
    STDOUT_LOG_FORMAT,
    JSON_DATE_LOG_FORMAT,
    JSON_LOG_FORMAT,
    JSON_LOG_OUTSUBDIR,
    JSON_LOG_OUTFILE,
    COLORED_FORMATTER_FIELD_STYLES,
    COLORED_FORMATTER_LEVEL_STYLES,
)
from .decorators import run_once


@dataclass
class JsonLoggingContext:
    out_directory: Path
    out_file_format: str
    log_format: str
    log_date_format: str


class MosplatLoggingBase:
    stdout_log_handler: ClassVar[logging.StreamHandler]
    json_log_handler: ClassVar[logging.StreamHandler]

    _root_logger: ClassVar[logging.Logger]

    class MosplatStdoutFormatter(coloredlogs.ColoredFormatter):
        def __init__(self, **kwargs):
            kwargs["field_styles"] = COLORED_FORMATTER_FIELD_STYLES
            kwargs["level_styles"] = COLORED_FORMATTER_LEVEL_STYLES
            super().__init__(**kwargs)

        def format(self, record: logging.LogRecord) -> str:
            style = self.nn.get(self.level_styles, record.levelname)
            # make custom level-aware record field that is first letter of log level
            if style:
                record.levelletter = ansi_wrap(
                    coerce_string(record.levelname[0]), **style
                )

            return super().format(record)

    @classmethod
    @run_once
    def init_once(cls, name: str):
        try:
            cls.set_log_record_factory()
            cls._root_logger = cls.configure_logger_instance(name)
            cls._root_logger.propagate = False  # prevent propogation to parent loggers

            cls.init_stdout_handler()

            cls._root_logger.info(f"Local root logger configured with name: `{name}`.")

        except Exception as e:
            # change default logger to the root logger as ours aren't setup properly
            cls._root_logger = logging.getLogger()
            cls._root_logger.exception(
                f"An exception occured while setting up logging: {e}"
            )

    @classmethod
    @run_once
    def init_stdout_handler(cls):
        """set formatter of stdout logger and add as handler"""

        if hasattr(cls, "stdout_log_handler"):
            return  # do not do anything if the handler already exists

        stdout_log_formatter = cls.MosplatStdoutFormatter(
            fmt=STDOUT_LOG_FORMAT, datefmt=STDOUT_DATE_LOG_FORMAT
        )
        cls.stdout_log_handler = logging.StreamHandler(sys.stdout)
        cls.stdout_log_handler.setFormatter(stdout_log_formatter)
        cls._root_logger.addHandler(cls.stdout_log_handler)

    @staticmethod
    @run_once
    def set_log_record_factory():
        """sets a custom log record factory"""
        old_factory = logging.getLogRecordFactory()

        def mosplat_record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.levelletter = record.levelname[0]
            record.dirname = os.path.basename(os.path.dirname(record.pathname))
            record.basename = record.name.rsplit(".", 1)[-1]
            return record

        logging.setLogRecordFactory(mosplat_record_factory)

    @classmethod
    def init_json_handler(cls, outdir: Path, file_fmt: str):
        """build path for json log output file, set formatter, and add as handler"""

        if hasattr(cls, "json_log_handler"):
            # if the handler already exists remove it and create new
            cls._root_logger.removeHandler(cls.json_log_handler)
            cls.json_log_handler.close()

        json_log_formatter = JsonFormatter(
            JSON_LOG_FORMAT, datefmt=JSON_DATE_LOG_FORMAT
        )

        os.makedirs(outdir, exist_ok=True)  # make the log directory if necessary
        json_log_outfile = os.path.join(
            outdir, datetime.datetime.now().strftime(file_fmt)
        )
        cls.json_log_handler = logging.FileHandler(json_log_outfile)
        cls.json_log_handler.setFormatter(json_log_formatter)

    @staticmethod
    def configure_logger_instance(name: str) -> logging.Logger:
        logger: logging.Logger = logging.getLogger(name)
        # set logger to most verbose
        logger.setLevel(logging.DEBUG)
        logger.propagate = True

        return logger

    @classmethod
    def cleanup(cls):
        """Remove handlers from the root logger."""
        if hasattr(cls, "stdout_log_handler"):
            cls._root_logger.removeHandler(cls.stdout_log_handler)
            cls.stdout_log_handler.close()
        if hasattr(cls, "json_log_handler"):
            cls._root_logger.removeHandler(cls.json_log_handler)
            cls.json_log_handler.close()
