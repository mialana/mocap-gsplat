from .main import register_addon, unregister_addon
from .infrastructure.logs import MosplatLoggingBase


def register():
    MosplatLoggingBase.init_once(
        __name__
    )  # initialize handlers and local "root" logger

    register_addon()


def unregister():
    unregister_addon()

    MosplatLoggingBase.cleanup()
