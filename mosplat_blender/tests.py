"""
some quick tests.
these defer from `.infrastructure.checks` as they are developer test that are not
actually called within the add-on's runtime.
"""

import os

from interfaces.logging_interface import LoggingInterface

logger = LoggingInterface.configure_logger_instance(__name__)


def test_deps_imports():
    try:
        import cv2
        import einops
        import huggingface_hub
        import numpy
        import PIL
        import plyfile
        import safetensors
        import torch
        import torchvision
        import vggt

        logger.info("Success! All dependencies could be imported.")
    except ImportError:
        logger.exception("Error importing a required dependency.")


def test_env():
    """currently just a convenience function to easily see env variables in debuggers."""
    env_list = []
    for k, v in os.environ.items():
        env_list.append((k, v))

    env_list.sort(key=lambda tup: tup[0])

    return
