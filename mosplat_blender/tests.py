"""
some quick tests.
these defer from `.infrastructure.checks` as they are developer test that are not
actually called within the add-on's runtime.
"""

import importlib
import os
from pathlib import Path
from typing import List

from .infrastructure.schemas import DeveloperError
from .interfaces.logging_interface import LoggingInterface

logger = LoggingInterface.configure_logger_instance(__name__)


def test_deps_imports():
    addon_dir = Path(__file__).resolve().parent
    requirements_txt_file = addon_dir / "requirements.txt"
    requirements_no_binary_txt_file = addon_dir / "requirements.no_binary.txt"

    requirements: List[str] = []

    with requirements_txt_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("--"):
                continue
            requirements.append(line.partition(";")[0])

    with requirements_no_binary_txt_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("--"):
                continue
            requirements.append(line.partition(";")[0])

    for req in requirements:
        try:
            importlib.import_module(req)
        except Exception as e:
            msg = DeveloperError.make_msg(
                f"Could not import required module: '{req}'", e
            )
            logger.error(msg)

    logger.info("Success! All dependencies could be imported.")


def test_env():
    """currently just a convenience function to easily see env variables in debuggers."""
    env_list = []
    for k, v in os.environ.items():
        env_list.append((k, v))

    env_list.sort(key=lambda tup: tup[0])

    return
