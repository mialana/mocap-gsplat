import bpy
import sys
import site
import subprocess
import threading
import importlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait

from .log_utils import logger

# List of packages required by the add-on/plugin
REQUIRED_PACKAGES = {
    "numpy": "numpy==1.26.1",
    "torch": "torch --index-url https://download.pytorch.org/whl/cu130",
    "torchvision": "torchvision --index-url https://download.pytorch.org/whl/cu130",
    "einops": "einops",
    "safetensors": "safetensors",
}


def ensure_pip():
    subprocess.check_call(
        [
            get_blender_python_path(),
            "-m",
            "ensurepip",
            "--upgrade",
        ]
    )


def get_blender_python_path():
    """Returns the path of Blender's embedded Python interpreter."""
    return sys.executable


def get_modules_path(path: str):
    """Return a writable directory for installing Python packages."""
    return bpy.utils.user_resource("SCRIPTS", path=path, create=True)


def append_modules_to_sys_path(modules_path):
    """Ensure Blender can find installed packages."""
    if modules_path not in sys.path:
        sys.path.append(modules_path)
    site.addsitedir(modules_path)


def display_message(message, title="Notification", icon="INFO"):
    """Show a popup message in Blender."""

    def draw(self, context):
        self.layout.label(text=message)

    def show_popup():
        bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)
        return None  # Stops timer

    bpy.app.timers.register(show_popup)


def install_package(module_name, pip_spec, modules_path):
    """Install a single package using Blender's Python."""
    try:
        print(f"Installing {pip_spec}...")

        list1 = [
            get_blender_python_path(),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--target",
            modules_path,
        ]
        list2 = pip_spec.split(" ")

        subprocess_list = list1 + list2
        subprocess.check_call(subprocess_list)

        print(f"{module_name} installed with spec '{pip_spec}' successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {pip_spec}. Error: {e}")
        display_message(
            f"Failed to install {pip_spec}. Check console for details.", icon="ERROR"
        )


def install_packages(packages, modules_path):
    wm = bpy.context.window_manager

    install_list = list(packages.items())[3:]
    wm.progress_begin(0, len(install_list))
    for i, (module_name, pip_spec) in enumerate(install_list):
        try:
            importlib.import_module(module_name)
            print(f"'{module_name}' is already installed.")
        except ImportError:
            install_package(module_name, pip_spec, modules_path)
        wm.progress_update(i + 1)
    wm.progress_end()
    display_message("All required packages installed successfully.")


def download_model():
    pass


def main():
    ensure_pip()
    modules_path = get_modules_path("modules/vggt_blender")
    append_modules_to_sys_path(modules_path)

    executor = ThreadPoolExecutor(max_workers=4)
    task1 = executor.submit(install_packages, REQUIRED_PACKAGES, modules_path)
    task2 = executor.submit(download_model)

    def check_done(_):
        if task1.done() and task2.done():
            print("done")

    task1.add_done_callback(check_done)
    task2.add_done_callback(check_done)


if __name__ == "__main__":
    main()
