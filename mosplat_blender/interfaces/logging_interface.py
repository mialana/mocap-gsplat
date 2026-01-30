"""
a static class to manage logging operations.
it contains custom formatter classes, handles handler creation,
sets the global `logging.LogRecordFactory`, and handles cleanup to not dirty the global namespace.
"""

from __future__ import annotations

import os
import sys
import logging
import datetime
from pathlib import Path
from typing import Callable, Optional, Self, ClassVar
from queue import Queue, Empty

from ..infrastructure.constants import (
    COLORED_FORMATTER_FIELD_STYLES,
    COLORED_FORMATTER_LEVEL_STYLES,
)
from ..infrastructure.protocols import SupportsMosplat_AP_Global
from ..infrastructure.decorators import run_once, run_once_per_instance
from ..infrastructure.schemas import UserFacingError, DeveloperError, OperatorIDEnum


class MosplatLoggingInterface:
    instance: ClassVar[Optional[Self]] = None
    instance_name: ClassVar[str]

    @run_once
    def __new__(cls, root_module_name: str) -> Self:
        """
        Expected usage is for only the top-level `__init__.py` of a package/program
        to create an instance of the interface.
        `root_module_name` should be the value of `__name__` at module-level.
        This allows `logging` to propogate handler and formatter settings to ALL
        other logging that occurs within the package.
        Proceeding attempts to create new instances will return that same interface.
        """

        cls.instance = super().__new__(cls)
        cls.instance_name = root_module_name
        return cls.instance

    @run_once_per_instance
    def __init__(self, root_module_name: str):
        self._root_logger: logging.Logger
        self._old_factory: Callable[..., logging.LogRecord]

        self._stdout_log_handler: logging.StreamHandler
        self._json_log_handler: logging.StreamHandler

        self._global_message_queue: Queue

        self._set_log_record_factory()

        self._root_logger = self.configure_logger_instance(root_module_name)
        self._root_logger.propagate = False  # prevent propogation to parent loggers

        self._root_logger.info(
            f"Local root logger configured with name: `{root_module_name}`."
        )

        self._global_message_queue = Queue()

    @classmethod
    def cleanup(cls):
        """Remove handlers from the root logger on singleton instance. reset instance."""
        self = cls.instance
        if self is None:
            return

        if self._stdout_log_handler:
            if self._root_logger:
                self._root_logger.removeHandler(self._stdout_log_handler)
            self._stdout_log_handler.close()
            del self._stdout_log_handler
        if self._json_log_handler:
            if self._root_logger:
                self._root_logger.removeHandler(self._json_log_handler)
            self._json_log_handler.close()
            del self._json_log_handler

        # drain message queue
        while True:
            try:
                self._global_message_queue.get_nowait()
            except Empty:
                break

        if self._old_factory:
            # restore old logrecord factory
            logging.setLogRecordFactory(self._old_factory)
            del self._old_factory

        cls.instance = None

    @run_once_per_instance
    def _set_log_record_factory(self):
        """sets a custom log record factory. should be only ran once, as"""
        self._old_factory = logging.getLogRecordFactory()

        def mosplat_record_factory(*args, **kwargs):
            if not self._old_factory:
                return logging.getLogRecordFactory()(*args, **kwargs)

            record: logging.LogRecord = self._old_factory(*args, **kwargs)
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

        return self._old_factory

    def _drain_global_message_queue(self):
        while not self._global_message_queue.empty():
            level, msg = self._global_message_queue.get_nowait()
            try:
                OperatorIDEnum.run(
                    OperatorIDEnum.REPORT_GLOBAL_MESSAGES,
                    "INVOKE_DEFAULT",
                    level=level,
                    message=msg,
                )
            except BaseException:
                pass

        return None

    @classmethod
    def init_handlers_from_addon_prefs(cls, addon_prefs: SupportsMosplat_AP_Global):
        self = cls.instance
        if not self:
            raise DeveloperError("Logging needs to be initialized from root module.")

        if self.init_stdout_handler(
            addon_prefs.stdout_log_format,
            addon_prefs.stdout_date_log_format,
        ):
            self._root_logger.info(
                f"STDOUT handler initialized from addon preferences."
            )

        if self.init_json_handler(
            log_fmt=addon_prefs.json_log_format,
            log_date_fmt=addon_prefs.json_date_log_format,
            outdir=Path(addon_prefs.cache_dir) / addon_prefs.json_log_subdir,
            file_fmt=addon_prefs.json_log_filename_format,
        ):
            self._root_logger.info(f"JSON handler initialized from addon preferences.")

    @classmethod
    def init_stdout_handler(cls, log_fmt: str, log_date_fmt: str) -> bool:
        """set formatter of stdout logger and add as handler"""
        self = cls.instance
        if not self:
            raise DeveloperError("Logging needs to be initialized from root module.")

        saved_handler = None
        if hasattr(self, "stdout_log_handler"):
            saved_handler = self._stdout_log_handler

        try:
            stdout_log_formatter = self.MosplatStdoutFormatter(
                fmt=log_fmt, datefmt=log_date_fmt
            )
            self._stdout_log_handler = logging.StreamHandler(sys.stdout)
            self._stdout_log_handler.setFormatter(stdout_log_formatter)
            self._root_logger.addHandler(self._stdout_log_handler)

            if saved_handler:
                # remove and close the old handler on success
                self._root_logger.removeHandler(saved_handler)
                saved_handler.close()

            return True
        except Exception as e:
            raise UserFacingError("Configuration for STDOUT handler invalid.", e) from e

    @classmethod
    def init_json_handler(
        cls, log_fmt: str, log_date_fmt: str, outdir: Path, file_fmt: str
    ) -> bool:
        """build path for json log output file, set formatter, and add as handler"""
        self = cls.instance
        if not self:
            raise DeveloperError("Logging needs to be initialized from root module.")

        if not self._root_logger:
            return False

        saved_handler = None
        if hasattr(self, "json_log_handler"):
            saved_handler = self._json_log_handler

        try:
            json_log_formatter = self.MosplatJsonFormatter(
                log_fmt, datefmt=log_date_fmt, json_indent=2  # indent 2 for readability
            )

            os.makedirs(outdir, exist_ok=True)  # make the log directory if necessary
            json_log_outfile = os.path.join(
                outdir, datetime.datetime.now().strftime(file_fmt)
            )
            self._json_log_handler = logging.FileHandler(json_log_outfile)
            self._json_log_handler.setFormatter(json_log_formatter)
            self._root_logger.addHandler(self._json_log_handler)

            if saved_handler:
                # remove and close the old handler on success
                self._root_logger.removeHandler(saved_handler)
                saved_handler.close()

            return True
        except Exception as e:
            raise UserFacingError("Configuration for JSON handler invalid.", e) from e

    @staticmethod
    def configure_logger_instance(name: str) -> logging.Logger:
        logger: logging.Logger = logging.getLogger(name)
        # set logger to most verbose
        logger.setLevel(logging.DEBUG)
        logger.propagate = True

        return logger

    # all non-standard lib modules should be nested locally

    import coloredlogs
    from pythonjsonlogger.json import JsonFormatter

    class MosplatStdoutFormatter(coloredlogs.ColoredFormatter):

        def __init__(self, **kwargs):
            kwargs["field_styles"] = COLORED_FORMATTER_FIELD_STYLES
            kwargs["level_styles"] = COLORED_FORMATTER_LEVEL_STYLES
            super().__init__(**kwargs)

        def format(self, record: logging.LogRecord) -> str:
            from humanfriendly.compat import coerce_string
            from humanfriendly.terminal import ansi_wrap

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
