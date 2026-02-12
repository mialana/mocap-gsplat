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
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))


from mosplat_blender.infrastructure.schemas import AddonMeta, EnvVariableEnum
from mosplat_blender.interfaces import LoggingInterface

ADDON_REGISTRATION_ID: Optional[str] = None


def register():
    resolve_addon_registration_id()
    setup_env()

    # delay import of `main` until after env setup
    from mosplat_blender.main import register_addon

    # initialize handlers and local "root" logger
    LoggingInterface(__name__)

    register_addon()


def unregister():
    from mosplat_blender.main import unregister_addon

    unregister_addon()

    LoggingInterface.cleanup_interface()

    if EnvVariableEnum.TESTING in os.environ:
        clear_terminal()  # for dev QOL

    for var in EnvVariableEnum:
        if var.value in os.environ:
            os.environ.pop(var.value, None)


def resolve_addon_registration_id():
    from bpy import context

    global ADDON_REGISTRATION_ID
    assert __package__  # running as an add-on guarantees `__package__` always exists
    assert context.preferences

    module_name: str = __package__.rpartition(".")[-1]

    # target last loaded
    for addon in reversed(context.preferences.addons.values()):
        if addon and module_name in addon.module:
            ADDON_REGISTRATION_ID = addon.module
            break


def setup_env():
    from dotenv import load_dotenv

    assert ADDON_REGISTRATION_ID

    AddonMeta(ADDON_REGISTRATION_ID)  # initialize global addon meta

    load_dotenv(Path(__file__).resolve().parent / ".production.env", verbose=True)

    # keep this module's name within env during execution
    os.environ.setdefault(EnvVariableEnum.ROOT_MODULE_NAME, __name__)
    os.environ.setdefault(EnvVariableEnum.ADDON_REGISTRATION_ID, ADDON_REGISTRATION_ID)


def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")
