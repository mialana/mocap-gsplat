import bpy
import sys
import os
import site
import subprocess
import importlib
import threading
import shutil

from . import constants
from . import helpers

from dataclasses import dataclass


@dataclass
class PackageSpecification:
    module_name: str
    pip_name: str
    pip_spec: str


REQUIRED_PACKAGES = [
    PackageSpecification(
        "numpy",
        "numpy",
        "numpy==1.26.1",
    ),
    PackageSpecification(
        "torch",
        "torch",
        "torch --index-url https://download.pytorch.org/whl/cu130",
    ),
    PackageSpecification(
        "torchvision",
        "torchvision",
        "torchvision --index-url https://download.pytorch.org/whl/cu130",
    ),
    PackageSpecification(
        "einops",
        "einops",
        "einops",
    ),
    PackageSpecification(
        "safetensors",
        "safetensors",
        "safetensors",
    ),
    PackageSpecification(
        "huggingface_hub",
        "huggingface_hub",
        "huggingface_hub",
    ),
    PackageSpecification(
        "git",
        "GitPython",
        "GitPython",
    ),
    PackageSpecification("plyfile", "plyfile", "plyfile"),
]


def get_blender_python_path():
    """Returns the path of Blender's embedded Python interpreter."""
    interpreter_path = sys.executable
    return interpreter_path


def ensure_pip():
    """Ensure pip is installed in Blender's Python environment."""
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
        raise


def append_to_path(modules_path):
    """Ensure Blender can find installed packages."""
    if modules_path not in sys.path:
        sys.path.append(modules_path)
    site.addsitedir(modules_path)
    print(f"{modules_path} added to PATH")


def ensure_package_installed(module_name):
    return importlib.util.find_spec(module_name) is not None


def install_package(module_name, pip_spec, modules_path):
    """Install a single package using Blender's Python."""
    print(f"Installing {pip_spec}...")

    try:
        subprocess_list = [
            get_blender_python_path(),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--quiet",
            "--upgrade",
            "--target",
            modules_path,
        ] + pip_spec.split(" ")

        subprocess.check_call(subprocess_list)
        print(f"{module_name} installed with spec '{pip_spec}' successfully.")

    except subprocess.CalledProcessError as e:
        print(f"Error installing {module_name}: {e}")
        raise


def install_vggt(modules_path):
    try:
        importlib.import_module("vggt")
        importlib.import_module("vggt.modules.vggt")
    except ImportError:
        git = importlib.import_module("git")

        wm = helpers.get_window_manager()
        wm.progress_begin(0, 2)  # start at 0, 2 steps

        out_dir: str = bpy.utils.user_resource(
            "SCRIPTS", path=constants.VGGT_REPO_SUBPATH, create=False
        )
        print(f"Installing VGGT to {out_dir}")

        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)

        try:
            git.Repo.clone_from(
                "https://github.com/facebookresearch/vggt.git",
                out_dir,
                multi_options=["--recurse-submodules"],
            )
        except git.GitCommandError as e:
            print(f"Error cloning VGGT from GitHub: {e}")
            raise e

        wm.progress_update(1)

        try:
            subprocess.check_call(
                [
                    get_blender_python_path(),
                    "-m",
                    "pip",
                    "install",
                    "--force-reinstall",
                    "--quiet",
                    "--upgrade",
                    "--target",
                    modules_path,
                    "-e",
                    out_dir,
                ]
            )
            wm.progress_update(2)
        except subprocess.CalledProcessError as e:
            print(f"Failed to install vggt as Python module. Error: {e}")
            raise
        wm.progress_end()

        append_to_path(out_dir)

    print("'vggt' is installed")
