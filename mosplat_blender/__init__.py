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
import sys
from pathlib import Path

from dotenv import load_dotenv

# save package before modifying path
ADDON_PACKAGE_ORIGINAL = __package__ or str(Path(__file__).resolve().parent)

sys.path.append(str(Path(__file__).resolve().parent))

from infrastructure.schemas import AddonMeta, EnvVariableEnum
from interfaces import MosplatLoggingInterface


def register():
    AddonMeta(ADDON_PACKAGE_ORIGINAL)  # initialize global addon meta
    setup_env()

    # delay import of `main` until after env setup
    from main import register_addon

    # initialize handlers and local "root" logger
    MosplatLoggingInterface(__name__)

    register_addon()


def unregister():
    from main import unregister_addon

    unregister_addon()

    MosplatLoggingInterface.cleanup_interface()

    if EnvVariableEnum.TESTING in os.environ:
        clear_terminal()  # for dev QOL

    os.environ.pop(EnvVariableEnum.ROOT_MODULE_NAME, None)


def setup_env():
    load_dotenv(Path(__file__).resolve().parent / ".production.env", verbose=True)

    # keep this module's name within env during execution
    os.environ.setdefault(EnvVariableEnum.ROOT_MODULE_NAME, __name__)
    os.environ.setdefault(
        EnvVariableEnum.ADDON_PACKAGE_ORIGINAL, ADDON_PACKAGE_ORIGINAL
    )


def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")
