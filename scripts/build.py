#!/usr/bin/env python3
"""
Author : Amy Liu <aliu@amyliu.dev>
Purpose:
1. Download PyPI wheels
2. Generate `blender_manifest.toml`
3. Package addon as ZIP file
4. Clean PyPI wheels directory (optional)
5. Install PyPI wheels to local Python interpreter for development (optional)
"""

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass, fields
from pathlib import Path
from string import capwords
from typing import Tuple

ADDON_HUMAN_READABLE = os.getenv("ADDON_HUMAN_READABLE", "mosplat_blender")

original_excepthook = sys.excepthook


def quiet_excepthook(exctype, value, traceback):
    # only print the value (error message) to the console
    print(f"Error: {value}", file=sys.stderr)


sys.excepthook = quiet_excepthook

step_tracker: int = 1


def _():
    """macro like function to separate print logs"""
    global step_tracker
    print(f"----------------- STEP{step_tracker}")
    step_tracker += 1


@dataclass
class BuildContext:
    """capitalized indicates it is an input resource (i.e. expected to exist)"""

    timestamp_str: str
    addon_base_id: str
    addon_human_readable: str
    version_tag: str
    wheels_dir: Path
    ADDON_SRC_DIR: Path
    ADDON_REQUIREMENTS_TXT_FILE: Path
    ADDON_NOBINARY_REQUIREMENTS_TXT_FILE: Path
    DEV_REQUIREMENTS_TXT_FILE: Path
    MANIFEST_TOML_TXT_FILE: Path
    manifest_toml_file: Path


def buildcontext_factory(scope) -> BuildContext:
    return BuildContext(**{f.name: scope[f.name] for f in fields(BuildContext)})


@dataclass
class ArgparseDefaults:
    """Argparse will override these defaults which in turn overrides build context"""

    addon_src_dir: Path


def get_args(defaults: ArgparseDefaults):
    parser = argparse.ArgumentParser(
        description="Purpose: 1. Download PyPI wheels 2. Generate `blender_manifest.toml` 3. Generate `.env`",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-a",
        "--addon_src_dir",
        help="Path to add-on source directory",
        type=Path,
        default=defaults.addon_src_dir,
    )

    parser.add_argument(
        "-v",
        "--blender_python_version",
        help="Version of Python interpreter within current Blender installation",
        type=str,
        default="3.11",
    )

    parser.add_argument(
        "-c",
        "--clean",
        help="Delete all wheels before downloading new wheels",
        action="store_true",
    )

    parser.add_argument(
        "-d",
        "--dev",
        help="Clean download PyPI wheels and install packages to current Python interpreter (development usage only)",
        action="store_true",
    )

    args = parser.parse_args()

    if args.dev:
        args.clean = True  # post-process args before return
    return args


def prepare_context() -> Tuple[BuildContext, argparse.Namespace]:
    """generate global build context using results of argparse."""
    repo_dir: Path = Path(__file__).resolve().parent.parent  # used in `.env`

    ADDON_SRC_DIR: Path = repo_dir / ADDON_HUMAN_READABLE  # used in `.env` and argparse

    defaults = ArgparseDefaults(addon_src_dir=ADDON_SRC_DIR)

    args = get_args(defaults)  # get program args from argparse

    ADDON_SRC_DIR: Path = args.addon_src_dir  # override defaults with new program args

    addon_base_id = ADDON_SRC_DIR.name
    addon_human_readable = capwords(addon_base_id.replace("_", " "))
    version_tag = get_version_tag_from_git()

    timestamp: datetime.datetime = datetime.datetime.now()
    timestamp_str = f"# {timestamp} \n# created using '{Path(__file__).name}'\n\n"

    wheels_dir: Path = ADDON_SRC_DIR / "wheels"
    print(f"Wheels Directory Path: {wheels_dir}")

    ADDON_REQUIREMENTS_TXT_FILE: Path = ADDON_SRC_DIR / "requirements.txt"

    print(f"Addon's `requirements.txt` File Path: {ADDON_REQUIREMENTS_TXT_FILE}")

    ADDON_NOBINARY_REQUIREMENTS_TXT_FILE: Path = (
        ADDON_SRC_DIR / "requirements.nobinary.txt"
    )

    print(
        f"Addon's `requirements.nobinary.txt` File Path: {ADDON_NOBINARY_REQUIREMENTS_TXT_FILE}"
    )

    DEV_REQUIREMENTS_TXT_FILE: Path = repo_dir / "requirements.txt"
    print(f"Developer's `requirements.txt` File Path: {DEV_REQUIREMENTS_TXT_FILE}")

    MANIFEST_TOML_TXT_FILE: Path = ADDON_SRC_DIR / "blender_manifest.toml.txt"
    print(f"Input `blender_manifest.txt` File Path: {MANIFEST_TOML_TXT_FILE}")

    manifest_toml_file: Path = ADDON_SRC_DIR / "blender_manifest.toml"
    print(f"Output `blender_manifest.toml` File Path: {manifest_toml_file}")

    return (
        buildcontext_factory(locals()),
        args,
    )


def get_version_tag_from_git() -> str:
    """Retreive git tag to use as canonical version truth for `blender_manifest.toml`."""
    try:
        git_version_tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            text=True,
        )
        print(f"Retrieved git version tag: {git_version_tag}")

        version_tag = git_version_tag.strip().removeprefix("v")
        print(f"Processed version tag for Blender manifest: {version_tag}")
        return version_tag
    except subprocess.CalledProcessError:
        print("No git tags found. Falling back to 0.0.0")
        return "0.0.0"


def install_dev_pypi_packages(ctx: BuildContext):
    """install wheels for development"""
    print("Beginning install...")

    pip_install_from_wheels_args_sublist = [  # shared between installation from wheels
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--only-binary=:all:",
        "--find-links",
        str(ctx.wheels_dir),
        "-r",
    ]
    try:
        subprocess.check_call(
            [
                *pip_install_from_wheels_args_sublist,
                str(ctx.ADDON_REQUIREMENTS_TXT_FILE),
            ]
        )
        subprocess.check_call(
            [
                *pip_install_from_wheels_args_sublist,
                str(ctx.ADDON_NOBINARY_REQUIREMENTS_TXT_FILE),
            ]
        )
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(ctx.DEV_REQUIREMENTS_TXT_FILE),
            ]
        )

    except subprocess.CalledProcessError as e:
        e.add_note(f"Error while installing PyPI wheels for development.")
        raise

    print("Install complete")


def download_pypi_wheels(ctx: BuildContext, blender_python_version, should_install):
    """use `subprocess` to invoke `pip download ...` on the addon's PyPI requirements"""
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "-r",
                str(ctx.ADDON_REQUIREMENTS_TXT_FILE),
                "--dest",
                str(ctx.wheels_dir),
                "--only-binary=:all:",
                f"--python-version={blender_python_version}",
            ]
        )
        # no-binary wheels
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "-r",
                str(ctx.ADDON_NOBINARY_REQUIREMENTS_TXT_FILE),
                "--wheel-dir",
                str(ctx.wheels_dir),
            ]
        )

    except subprocess.CalledProcessError as e:
        e.add_note(f"Error in call to download PyPI wheels.")
        raise

    print("All PyPI wheels successfully downloaded.")

    if should_install:
        _()  # skip a line
        install_dev_pypi_packages(ctx)


def generate_blender_manifest_toml(ctx: BuildContext):
    """
    append the downloaded `*.whl` files to the template `blender_manifest.txt file
    & save as `blender_manifest.toml`.
    """

    print(f"`Generating `blender_manifest.toml`...")
    wheels = sorted(
        f'  "./wheels/{f}"' for f in os.listdir(ctx.wheels_dir) if f.endswith(".whl")
    )
    WHEELS_STR = "wheels = [\n" + ",\n".join(wheels) + "\n]"

    with open(ctx.MANIFEST_TOML_TXT_FILE, "r", encoding="utf-8") as f:
        next(f)  # skip the comment on the first line
        template = f.read()

    template = template.format(
        addon_base_id=ctx.addon_base_id,
        addon_human_readable=ctx.addon_human_readable,
        version_tag=ctx.version_tag,
        wheels_block=WHEELS_STR,
    )

    with open(ctx.manifest_toml_file, "w", encoding="utf-8") as f:
        f.write(ctx.timestamp_str)
        f.write(template)

    print(f"`{ctx.manifest_toml_file}` successfully generated.")


def package(ctx: BuildContext):
    zip_path = ctx.ADDON_SRC_DIR.parent / f"{ctx.ADDON_SRC_DIR.name}.zip"

    if zip_path.exists():
        zip_path.unlink()

    ignore_dirs = {"__pycache__"}
    ignore_extensions = {".zip"}
    ignore_filenames = {"blender_manifest.toml.txt"}

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for root, dirs, files in os.walk(ctx.ADDON_SRC_DIR):
            # mutate dirs in-place to prevent descending
            dirs[:] = [d for d in dirs if d not in ignore_dirs]

            for file in files:
                path = Path(root) / file

                if file in ignore_filenames or path.suffix in ignore_extensions:
                    continue

                path_in_archive = path.relative_to(ctx.ADDON_SRC_DIR.parent)
                zf.write(path, path_in_archive)

    print(f"Addon packaged as `{zip_path}`")


def clean(wheels_dir):
    print(f"Beginning clean...")
    if os.path.exists(wheels_dir) and os.path.isdir(wheels_dir):
        try:
            shutil.rmtree(wheels_dir)
            print(f"Wheels directory and all its contents have been removed.")
        except OSError:
            print(f"Error deleting wheels directory")
            raise

    print(f"Clean complete.")


def main():
    _()  # skip a line
    ctx, args = prepare_context()

    for p in [
        ctx.ADDON_SRC_DIR,
        ctx.ADDON_REQUIREMENTS_TXT_FILE,
        ctx.ADDON_NOBINARY_REQUIREMENTS_TXT_FILE,
        ctx.MANIFEST_TOML_TXT_FILE,
        *([ctx.DEV_REQUIREMENTS_TXT_FILE] if args.dev else []),
    ]:  # check that all required input resources are where they are supposed to be
        if not p.exists():
            raise RuntimeError(f"Expected {p!r} in file system, but was not found.")

    _()  # skip a line

    if args.clean:
        clean(ctx.wheels_dir)
        _()  # skip a line

    # download_pypi_wheels(ctx, args.blender_python_version, should_install=args.dev)
    _()  # skip a line

    generate_blender_manifest_toml(ctx)

    package(ctx)


if __name__ == "__main__":
    main()
