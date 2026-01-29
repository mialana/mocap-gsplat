"""
a static class to manage logging operations.
it contains custom formatter classes, handles handler creation,
sets the global `logging.LogRecordFactory`, and handles cleanup to not dirty the global namespace.
"""

import os
import sys
import logging
import datetime
from pathlib import Path
from typing import ClassVar, Callable, final
from pythonjsonlogger.json import JsonFormatter
import coloredlogs
from humanfriendly.compat import coerce_string
from humanfriendly.terminal import ansi_wrap
from ..infrastructure.constants import (
    COLORED_FORMATTER_FIELD_STYLES,
    COLORED_FORMATTER_LEVEL_STYLES,
)
from ..infrastructure.protocols import SupportsMosplat_AP_Global
from ..infrastructure.decorators import run_once, no_instantiate
from ..infrastructure.schemas import UserFacingError


@final
@no_instantiate
class MosplatLoggingInterface:
    stdout_log_handler: ClassVar[logging.StreamHandler | None] = None
    json_log_handler: ClassVar[logging.StreamHandler | None] = None

    _root_logger: ClassVar[logging.Logger | None] = None

    _old_factory: ClassVar[Callable[..., logging.LogRecord] | None] = None

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

    class MosplatJsonFormatter(JsonFormatter):
        def process_log_record(self, log_data: dict):
            """remove custom logrecord attribs that are unnecessary for json logging"""
            log_data.pop("levelletter", None)
            log_data.pop("dirname", None)
            return log_data

    @classmethod
    @run_once
    def init_once(cls, name: str):
        cls.set_log_record_factory()
        cls._root_logger = cls.configure_logger_instance(name)
        cls._root_logger.propagate = False  # prevent propogation to parent loggers

        cls._root_logger.info(f"Local root logger configured with name: `{name}`.")

    @classmethod
    def init_handlers_from_addon_prefs(cls, addon_prefs: SupportsMosplat_AP_Global):
        if not cls._root_logger:
            return

        if cls.init_stdout_handler(
            addon_prefs.stdout_log_format,
            addon_prefs.stdout_date_log_format,
        ):
            cls._root_logger.info(f"STDOUT handler initialized from addon preferences.")

        if cls.init_json_handler(
            log_fmt=addon_prefs.json_log_format,
            log_date_fmt=addon_prefs.json_date_log_format,
            outdir=Path(addon_prefs.cache_dir) / addon_prefs.json_log_subdir,
            file_fmt=addon_prefs.json_log_filename_format,
        ):
            cls._root_logger.info(f"JSON handler initialized from addon preferences.")

    @classmethod
    @run_once
    def cleanup(cls):
        """Remove handlers from the root logger."""
        if cls.stdout_log_handler:
            if cls._root_logger:
                cls._root_logger.removeHandler(cls.stdout_log_handler)
            cls.stdout_log_handler.close()
            cls.stdout_log_handler = None
        if cls.json_log_handler:
            if cls._root_logger:
                cls._root_logger.removeHandler(cls.json_log_handler)
            cls.json_log_handler.close()
            cls.json_log_handler = None

        if cls._old_factory:
            logging.setLogRecordFactory(
                cls._old_factory
            )  # restore old logrecord factory
            cls._old_factory = None
        cls._root_logger = None

    @classmethod
    def init_stdout_handler(cls, log_fmt: str, log_date_fmt: str) -> bool:
        """set formatter of stdout logger and add as handler"""
        if not cls._root_logger:
            return False

        saved_handler = None
        if hasattr(cls, "stdout_log_handler"):
            saved_handler = cls.stdout_log_handler

        try:
            stdout_log_formatter = cls.MosplatStdoutFormatter(
                fmt=log_fmt, datefmt=log_date_fmt
            )
            cls.stdout_log_handler = logging.StreamHandler(sys.stdout)
            cls.stdout_log_handler.setFormatter(stdout_log_formatter)
            cls._root_logger.addHandler(cls.stdout_log_handler)

            if saved_handler:
                # remove and close the old handler on success
                cls._root_logger.removeHandler(saved_handler)
                saved_handler.close()

            return True
        except Exception as e:
            raise UserFacingError("Configuration for STDOUT handler invalid.", e) from e

    @classmethod
    def init_json_handler(
        cls, log_fmt: str, log_date_fmt: str, outdir: Path, file_fmt: str
    ) -> bool:
        """build path for json log output file, set formatter, and add as handler"""
        if not cls._root_logger:
            return False

        saved_handler = None
        if hasattr(cls, "json_log_handler"):
            saved_handler = cls.json_log_handler

        try:
            json_log_formatter = cls.MosplatJsonFormatter(
                log_fmt, datefmt=log_date_fmt, json_indent=2  # indent 2 for readability
            )

            os.makedirs(outdir, exist_ok=True)  # make the log directory if necessary
            json_log_outfile = os.path.join(
                outdir, datetime.datetime.now().strftime(file_fmt)
            )
            cls.json_log_handler = logging.FileHandler(json_log_outfile)
            cls.json_log_handler.setFormatter(json_log_formatter)
            cls._root_logger.addHandler(cls.json_log_handler)

            if saved_handler:
                # remove and close the old handler on success
                cls._root_logger.removeHandler(saved_handler)
                saved_handler.close()

            return True
        except Exception as e:
            raise UserFacingError("Configuration for JSON handler invalid.", e) from e

    @classmethod
    def set_log_record_factory(cls):
        """sets a custom log record factory"""
        if cls._old_factory:
            return

        cls._old_factory = logging.getLogRecordFactory()

        def mosplat_record_factory(*args, **kwargs):
            if not cls._old_factory:
                return logging.getLogRecordFactory()(*args, **kwargs)

            record = cls._old_factory(*args, **kwargs)
            record.levelletter = record.levelname[0]  # parse for just the first letter
            record.dirname = os.path.basename(os.path.dirname(record.pathname))
            record.classname = (
                record.name.split("logclass")[-1]
                if "logclass" in record.name
                else "<noclass>"
            )  # get the pre-defined `__qualname__` part of the name, indicating it is a `MosplatLogClassMixin` subclass
            if record.funcName == "<module>":
                record.funcName = "<nofunction>"  # to match the above pattern
            return record

        logging.setLogRecordFactory(mosplat_record_factory)

    @staticmethod
    def configure_logger_instance(name: str) -> logging.Logger:
        logger: logging.Logger = logging.getLogger(name)
        # set logger to most verbose
        logger.setLevel(logging.DEBUG)
        logger.propagate = True

        return logger
