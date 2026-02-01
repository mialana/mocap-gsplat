"""
a class to manage logging operations. it is meant to be used as a global singleton.
it contains custom formatter classes,
handles handler creation,
sets the global `logging.LogRecordFactory`,
and handles cleanup to not dirty the global namespace.
"""

from __future__ import annotations

import sys
import logging
import datetime
from pathlib import Path
from typing import Callable, Optional, Self, ClassVar, NoReturn, Tuple, TypeAlias
from queue import Queue, Empty
from enum import StrEnum, auto

from ..infrastructure.constants import (
    COLORED_FORMATTER_FIELD_STYLES,
    COLORED_FORMATTER_LEVEL_STYLES,
    MAX_LOG_ENTRIES_STORED,
)
from ..infrastructure.protocols import SupportsMosplat_AP_Global
from ..infrastructure.decorators import run_once_per_instance, run_once
from ..infrastructure.schemas import (
    UserFacingError,
    DeveloperError,
    UnexpectedError,
    LogEntryLevelEnum,
)

BlenderLogEntry: TypeAlias = Tuple[LogEntryLevelEnum, str, str]


class LoggingHandler(StrEnum):
    STDOUT = auto()
    JSON = auto()
    BLENDER = auto()


class MosplatLoggingInterface:
    instance: ClassVar[Optional[Self]]
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

        self._stdout_log_handler: Optional[logging.StreamHandler] = None
        self._json_log_handler: Optional[logging.FileHandler] = None
        self._blender_report_handler: Optional[MosplatBlenderReportHandler] = None

        self._blender_log_entry_queue: Queue[BlenderLogEntry]

        self._set_log_record_factory()

        self._root_logger = self.configure_logger_instance(root_module_name)
        self._root_logger.propagate = False  # prevent propogation to parent loggers

        self._blender_log_entry_queue = Queue()

    @classmethod
    def cleanup(cls):
        "cleanup instance. reset instance."
        if cls.instance is None:
            return
        cls.instance._cleanup()
        cls.instance = None

    @staticmethod
    def configure_logger_instance(name: str) -> logging.Logger:
        logger: logging.Logger = logging.getLogger(name)
        # set logger to most verbose
        logger.setLevel(logging.DEBUG)
        logger.propagate = True

        logger.debug(f"Configured logger instance '{name}'.")

        return logger

    @classmethod
    def init_handlers_from_addon_prefs(cls, addon_prefs: SupportsMosplat_AP_Global):
        if cls.instance is None:
            cls.error_out()

        cls.instance._init_handlers_from_addon_prefs(addon_prefs)

    @classmethod
    def init_stdout_handler(cls, log_fmt: str, log_date_fmt: str) -> bool:
        """set formatter of stdout logger and add as handler"""
        if cls.instance is None:
            cls.error_out()

        return cls.instance._init_stdout_handler(log_fmt, log_date_fmt)

    @classmethod
    def init_json_handler(
        cls, log_fmt: str, log_date_fmt: str, outdir: Path, file_fmt: str
    ) -> bool:
        """build path for json log output file, set formatter, and add as handler"""
        if cls.instance is None:
            cls.error_out()

        return cls.instance._init_json_handler(log_fmt, log_date_fmt, outdir, file_fmt)

    @classmethod
    def add_global_message(cls, msg: BlenderLogEntry):
        if cls.instance is None:
            cls.error_out()

        cls.instance._add_blender_log_entry(msg)

    @classmethod
    def error_out(cls) -> NoReturn:
        raise DeveloperError("Logging needs to be initialized from root module.")

    def _init_handlers_from_addon_prefs(self, prefs: SupportsMosplat_AP_Global):
        logger = self._root_logger

        ok_fmt = "'{handler_type}' logger initialized from addon prefs."
        error_fmt = "'{handler_type}' logger cannot be initialized from addon prefs."

        for handler in LoggingHandler:
            ok = False
            msg = ok_fmt
            try:
                ok = self._init_handler(handler, prefs)
            except UserFacingError as e:
                msg = UserFacingError.make_msg(error_fmt, e)

            log_msg = msg.format(handler_type=handler.upper())
            logger.info(log_msg) if ok else logger.warning(log_msg)

    def _init_handler(
        self, handler: LoggingHandler, prefs: SupportsMosplat_AP_Global
    ) -> bool:
        """internal convenience function"""
        match handler:
            case LoggingHandler.STDOUT:
                return self._init_stdout_handler(
                    prefs.stdout_log_format, prefs.stdout_date_log_format
                )
            case LoggingHandler.JSON:
                return self._init_json_handler(
                    log_fmt=prefs.json_log_format,
                    log_date_fmt=prefs.json_date_log_format,
                    outdir=Path(prefs.cache_dir) / prefs.json_log_subdir,
                    file_fmt=prefs.json_log_filename_format,
                )
            case LoggingHandler.BLENDER:
                return self._init_blender_report_handler()

    def _init_stdout_handler(self, log_fmt: str, log_date_fmt: str) -> bool:
        """set formatter of stdout logger and add as handler"""
        saved_handler: Optional[logging.StreamHandler] = None
        if self._stdout_log_handler is not None:
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
            raise UserFacingError("Config for STDOUT handler failed.", e) from e

    def _init_json_handler(
        self, log_fmt: str, log_date_fmt: str, outdir: Path, file_fmt: str
    ) -> bool:
        """build path for json log output file, set formatter, and add as handler"""
        saved_handler: Optional[logging.FileHandler] = None
        if self._json_log_handler is not None:
            saved_handler = self._json_log_handler

        try:
            json_log_formatter = self.MosplatJsonFormatter(
                log_fmt, datefmt=log_date_fmt, json_indent=2  # indent 2 for readability
            )
            outdir.mkdir(exist_ok=True)  # make the log directory if necessary

            json_log_outfile = outdir / datetime.datetime.now().strftime(file_fmt)
            self._json_log_handler = logging.FileHandler(json_log_outfile)
            self._json_log_handler.setFormatter(json_log_formatter)
            self._root_logger.addHandler(self._json_log_handler)

            if saved_handler:
                # remove and close the old handler on success
                self._root_logger.removeHandler(saved_handler)
                saved_handler.close()

            return True
        except Exception as e:
            raise UserFacingError("Configuration for JSON handler failed.", e) from e

    def _init_blender_report_handler(self) -> bool:
        saved_handler: Optional[MosplatBlenderReportHandler] = None
        if self._blender_report_handler is not None:
            saved_handler = self._blender_report_handler
        try:
            self._blender_report_handler = MosplatBlenderReportHandler()
            self._root_logger.addHandler(self._blender_report_handler)

            if saved_handler:
                # remove and close the old handler on success
                self._root_logger.removeHandler(saved_handler)
                saved_handler.close()
            return True
        except Exception as e:
            raise UserFacingError("Config for Blender handler failed.", e) from e

    @run_once_per_instance
    def _set_log_record_factory(self):
        """
        sets a custom log record factory.
        should be only ran once, as otherwise the old log record is corrupted.
        """
        self._old_factory = logging.getLogRecordFactory()

        def mosplat_record_factory(*args, **kwargs):
            if not self._old_factory:
                return logging.getLogRecordFactory()(*args, **kwargs)

            record: logging.LogRecord = self._old_factory(*args, **kwargs)
            record.levelletter = record.levelname[0]  # parse for just the first letter
            record.dirname = Path(record.pathname).parent.name
            record.classname = (
                record.name.split("logclass")[-1]
                if "logclass" in record.name
                else "<noclass>"
            )  # get the pre-defined `__name__` part of the name, indicating it is a `LogClassMixin` subclass
            if record.funcName == "<module>":
                record.funcName = "<nofunction>"  # to match the above pattern
            return record

        logging.setLogRecordFactory(mosplat_record_factory)

        return self._old_factory

    def _cleanup(self):
        # drain message queue
        try:
            self._drain_blender_log_entry_queue()
        except (UnexpectedError, UserFacingError):
            pass  # cleaning up so this is expected

        """remove handlers from the root logger"""
        if self._stdout_log_handler:
            if self._root_logger:
                self._root_logger.removeHandler(self._stdout_log_handler)
            self._stdout_log_handler.close()
        if self._json_log_handler:
            if self._root_logger:
                self._root_logger.removeHandler(self._json_log_handler)
            self._json_log_handler.close()
        if self._blender_report_handler:
            if self._root_logger:
                self._root_logger.removeHandler(self._blender_report_handler)
            self._blender_report_handler.close()

        if self._old_factory:
            # restore old logrecord factory
            logging.setLogRecordFactory(self._old_factory)

    def _drain_blender_log_entry_queue(self):
        """pops items from local queue and adds to Blender collection property"""
        from bpy import context  # local runtime import for most up-to-date context
        from ..core.checks import check_propertygroup

        try:
            log_hub = check_propertygroup(context.scene).log_hub_accessor

            entries = log_hub.entries_accessor

            while not self._blender_log_entry_queue.empty():
                level, msg, full_msg = self._blender_log_entry_queue.get_nowait()

                entry = entries.add()
                entry.level = level.value
                entry.message = msg
                entry.full_message = full_msg

                while len(entries) > MAX_LOG_ENTRIES_STORED:
                    entries.remove(0)

                log_hub.logs_active_index = len(entries) - 1

        except (UnexpectedError, UserFacingError) as e:
            e.add_note(f"NOTE: caught while draining blender log entry queue.")
            raise

    def _add_blender_log_entry(self, entry: BlenderLogEntry):
        from bpy.app import timers  # local import

        self._blender_log_entry_queue.put(entry)
        timers.register(self._drain_blender_log_entry_queue, first_interval=0)

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


class MosplatBlenderReportHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            classname = getattr(record, "classname", None)
            asctime = getattr(record, "asctime", None)
            thread = getattr(record, "thread", None)

            full_msg = msg
            if classname:
                full_msg += f"\n{classname=}"
            if asctime:
                full_msg += f"\n{asctime=}"
            if thread:
                full_msg += f"\n{thread=}"

            full_msg += "\n"  # skip a line

            levelname = record.levelname
            item = LogEntryLevelEnum.from_log_record(levelname)

            MosplatLoggingInterface.add_global_message((item, msg, full_msg))

        except Exception:
            # logging handlers should never raise
            self.handleError(record)
