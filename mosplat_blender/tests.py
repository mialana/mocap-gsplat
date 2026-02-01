"""
some quick tests.
these defer from `.infrastructure.checks` as they are developer test that are not
actually called within the add-on's runtime.
"""

from .interfaces.logging_interface import MosplatLoggingInterface

logger = MosplatLoggingInterface.configure_logger_instance(__name__)


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
