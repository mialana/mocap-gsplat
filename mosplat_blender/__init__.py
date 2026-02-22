"""
defers registration and unregistration to `main.py`, following native Python practices to keep implementation logic out of `__init__.py`.

defers registration and unregistration to `main.py`, following native Python practices to keep implementation logic out of `__init__.py`.

it does however initialize a local root logger using the module's name, which allows the created logger to become the local "root logger" that other loggers in the addon will propogate from (i.e. inherit handlers and formatters).
this is because propogation hierarchies  in the `logging` module follows dot notation, so this module is essentially `A`, other files in this directory like `main.py` as well as subdirectories' `__init__.py`'s are `A.B`, files within subdirectories are `A.B.C`, and so on (ref: https://docs.python.org/3/library/logging.html#logging.Logger.propagate)
"""

import os
import sys
from pathlib import Path
from typing import Optional

from .infrastructure.schemas import AddonMeta, EnvVariableEnum
from .interfaces.logging_interface import LoggingInterface

ADDON_REGISTRATION_ID: Optional[str] = None
SUBPROC_IMPORT_ROOT_DIR: Optional[str] = None


def register():
    resolve_addon_registration_id()
    setup_env()

    # initialize handlers and local "root" logger
    LoggingInterface(__name__)

    # delay import of `main` until after env setup
    from .main import register_addon

    register_addon()


def unregister():
    from .main import unregister_addon

    should_clear = True
    try:
        unregister_addon()
    except Exception as e:
        if LoggingInterface.instance:
            LoggingInterface.instance._root_logger.error(str(e))
        should_clear = False

    LoggingInterface.cleanup_interface()

    if EnvVariableEnum.TESTING in os.environ and should_clear:
        clear_terminal()  # for dev QOL

    for var in EnvVariableEnum:
        if var.value in os.environ:
            os.environ.pop(var.value, None)

    if SUBPROC_IMPORT_ROOT_DIR and SUBPROC_IMPORT_ROOT_DIR in sys.path:
        sys.path.remove(SUBPROC_IMPORT_ROOT_DIR)  # all clean!


def resolve_addon_registration_id():
    from bpy import context

    global ADDON_REGISTRATION_ID
    assert context.preferences

    base_module_name: str = __name__.rpartition(".")[-1]

    # target last loaded
    for addon in reversed(context.preferences.addons.values()):
        if addon and base_module_name in addon.module:
            ADDON_REGISTRATION_ID = addon.module
            break


def setup_env():
    from dotenv import load_dotenv

    global SUBPROC_IMPORT_ROOT_DIR

    assert ADDON_REGISTRATION_ID

    meta = AddonMeta(ADDON_REGISTRATION_ID)  # initialize global addon meta

    load_dotenv(Path(__file__).resolve().parent / ".production.env", verbose=True)

    # keep this module's name within env during execution
    os.environ.setdefault(EnvVariableEnum.ROOT_MODULE_NAME, __name__)
    os.environ.setdefault(EnvVariableEnum.ADDON_REGISTRATION_ID, ADDON_REGISTRATION_ID)

    # subprocess will resolve imports from the addon's parent directory. we need to insert into `sys.path` to ensure the import will work
    SUBPROC_IMPORT_ROOT_DIR = str(meta.addon_parent_dir)
    if SUBPROC_IMPORT_ROOT_DIR not in sys.path:
        sys.path.insert(0, SUBPROC_IMPORT_ROOT_DIR)


def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")
