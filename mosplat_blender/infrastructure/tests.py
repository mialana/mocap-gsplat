"""Some quick tests"""

from .interfaces.logging_interface import MosplatLoggingInterface

logger = MosplatLoggingInterface.configure_logger_instance(__name__)


def test_deps_imports():
    try:
        import numpy
        import PIL
        import huggingface_hub
        import einops
        import safetensors
        import plyfile
        import torch
        import torchvision
        import vggt

        logger.info("Success! All dependencies could be imported.")
    except ImportError:
        logger.exception("Error importing a required dependency.")
