"""
defers registration and unregistration to `main.py`,
following native Python practices to keep implementation logic out of `__init__.py`.

it does however call the `init_once` function of the logging interface,
which allows the created logger to become the local "root logger" that other
loggers in the addon will propogate from (i.e. inherit handlers and formatters).
this is because propogation hierarchies  in the `logging` module follows dot notation,
so this module is essentially `A`, other files in this directory like `main.py`
as well as subdirectories' `__init__.py`'s are `A.B`, files within subdirectories
are `A.B.C`, and so on (ref: https://docs.python.org/3/library/logging.html#logging.Logger.propagate)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from .main import register_addon, unregister_addon
from .interfaces import MosplatLoggingInterface


def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")


def register():
    load_dotenv(
        Path(__file__).resolve().parent.joinpath(".production.env"), verbose=True
    )

    # initialize handlers and local "root" logger
    MosplatLoggingInterface.init_once(__name__)

    register_addon()


def unregister():
    unregister_addon()

    MosplatLoggingInterface.cleanup()
    clear_terminal()  # for dev QOL
