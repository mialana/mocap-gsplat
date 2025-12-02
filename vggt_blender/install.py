import bpy
import sys
import site
import subprocess
import importlib
import threading
from concurrent.futures import ThreadPoolExecutor

from . import constants
from . import helpers


# List of packages required by the add-on/plugin
REQUIRED_PACKAGES = {
    "numpy": "numpy==1.26.1",
    "torch": "torch --index-url https://download.pytorch.org/whl/cu130",
    "torchvision": "torchvision --index-url https://download.pytorch.org/whl/cu130",
    "einops": "einops",
    "safetensors": "safetensors",
    "huggingface_hub": "huggingface_hub",
    "GitPython": "GitPython",
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


def append_modules_to_sys_path(modules_path):
    """Ensure Blender can find installed packages."""
    if modules_path not in sys.path:
        sys.path.append(modules_path)
    site.addsitedir(modules_path)


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
        print(f"Failed to install {pip_spec}. Error: {e}")
        helpers.display_message(
            f"Failed to install {pip_spec}. Check console for details.", icon="ERROR"
        )


def install_packages(packages, modules_path):
    install_list = list(packages.items())

    wm = bpy.context.window_manager
    wm.progress_begin(0, len(install_list))
    for i, (module_name, pip_spec) in enumerate(install_list):
        installed: bool = False
        while not installed:
            try:
                importlib.import_module(module_name)
                print(f"'{module_name}' is already installed.")
                installed = True
            except ImportError:
                install_package(module_name, pip_spec, modules_path)
        wm.progress_update(i + 1)
    wm.progress_end()
    helpers.display_message("All required packages installed successfully.")


def install_vggt():
    git = importlib.import_module("git")

    wm = helpers.get_window_manager()
    wm.progress_begin(0, 2)  # start at 0, 2 steps

    out_dir: str = bpy.utils.user_resource(
        "SCRIPTS", path=constants.VGGT_REPO_SUBPATH, create=True
    )
    git.Repo.clone_from("https://github.com/facebookresearch/vggt.git", out_dir)

    wm.progress_update(1)

    try:
        subprocess.check_call(["python", "-m" "pip", "install", "-e", out_dir])
        wm.progress_update(2)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install vggt as Python module. Error: {e}")
        helpers.display_message(
            f"Failed to install vggt as Python module. Check console for details.",
            icon="ERROR",
        )
    wm.progress_end()


def background_installation(modules_path):
    install_packages(REQUIRED_PACKAGES, modules_path)
    install_vggt()


def main():
    ensure_pip()
    modules_path = helpers.resolve_script_file_path(constants.ADDON_SUBPATH)

    append_modules_to_sys_path(modules_path)

    threading.Thread(
        target=background_installation, daemon=True, args=(modules_path,)
    ).start()


if __name__ == "__main__":
    main()
