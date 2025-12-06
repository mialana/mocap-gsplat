import bpy
import sys
import os
import site
import subprocess
import importlib
import threading
from concurrent.futures import ThreadPoolExecutor

from . import constants
from . import helpers

print(sys.executable)

# List of packages required by the add-on/plugin
REQUIRED_PACKAGES = {
    "numpy": "numpy==1.26.1",
    "torch": "torch --index-url https://download.pytorch.org/whl/cu130",
    "torchvision": "torchvision --index-url https://download.pytorch.org/whl/cu130",
    "einops": "einops",
    "safetensors": "safetensors",
    "huggingface_hub": "huggingface_hub",
    "git": "GitPython",
}


def get_blender_python_path():
    """Returns the path of Blender's embedded Python interpreter."""
    return sys.executable


def ensure_pip():
    try:
        subprocess.check_call(
            [
                get_blender_python_path(),
                "-m",
                "ensurepip",
                "--upgrade",
            ]
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to install pip. Error: {e}")
        helpers.display_message(
            f"Failed to install pip. Check console for details.", icon="ERROR"
        )
        raise


def append_modules_to_sys_path(modules_path):
    """Ensure Blender can find installed packages."""
    if modules_path not in sys.path:
        sys.path.append(modules_path)
    site.addsitedir(modules_path)


def install_package(module_name, pip_spec, modules_path):
    """Install a single package using Blender's Python."""
    print(f"Installing {pip_spec}...")

    subprocess_list = [
        get_blender_python_path(),
        "-m",
        "pip",
        "install",
        "--quiet",
        "--disable-pip-version-check",
        "--upgrade",
        "--target",
        modules_path,
    ] + pip_spec.split(" ")

    subprocess.check_call(subprocess_list)

    print(f"{module_name} installed with spec '{pip_spec}' successfully.")


def install_packages(packages, modules_path):
    install_list = list(packages.items())

    wm = bpy.context.window_manager
    wm.progress_begin(0, len(install_list))

    for i, (module_name, pip_spec) in enumerate(install_list):
        installed: bool = False
        try:
            importlib.import_module(module_name)
            installed = True
        except ImportError:
            try:
                install_package(module_name, pip_spec, modules_path)
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {pip_spec}. Error: {e}")
                helpers.display_message(
                    f"Failed to install {pip_spec}. Check console for details.",
                    icon="ERROR",
                )
                raise

        pkg_path = os.path.join(modules_path, module_name)
        if not installed and os.path.exists(pkg_path):
            installed = True

        print(f"'{module_name}' is installed.")
        wm.progress_update(i + 1)

    wm.progress_end()
    helpers.display_message("All required packages installed successfully.")


def install_vggt():
    try:
        importlib.import_module("vggt")
    except ImportError:
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
            raise
        wm.progress_end()
    
    print("'vggt' is installed")


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
