#!/usr/bin/env python3
"""
Author : Amy Liu <aliu@amyliu.dev>
Purpose:
1. Download PyPI wheels
2. Generate `blender_manifest.toml`
3. Generate `.env`
"""

import argparse
from pathlib import Path
import sys
import os
import shutil
import subprocess
import datetime
from typing import Protocol, Tuple, Any
from dataclasses import dataclass


@dataclass
class BuildContext:
    """capitalized indicates it is an input resource (i.e. expected to exist)"""

    timestamp_str: str
    wheels_dir: Path
    REQUIREMENTS_TXT_FILE: Path
    MANIFEST_TXT_FILE: Path
    manifest_toml_file: Path
    DOTENV_TXT_FILE: Path
    dotenv_file: Path
    ENVVAR_REPO_DIR: Path
    ENVVAR_ADDON_SRC_DIR: Path
    envvar_cache_dir: Path


@dataclass
class ArgparseDefaults:
    addon_src_dir: Path  # overrides build context


def get_args(defaults: ArgparseDefaults):
    parser = argparse.ArgumentParser(
        description="Purpose: 1. Download PyPI wheels 2. Generate `blender_manifest.toml` 3. Generate `.env`",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-a",
        "--addon_src_dir",
        help="Path to Mosplat Blender add-on directory",
        type=Path,
        default=defaults.addon_src_dir,
    )

    parser.add_argument(
        "-v",
        "--version",
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
    ENVVAR_REPO_DIR: Path = Path(__file__).resolve().parent.parent  # used in `.env`
    envvar_cache_dir: Path = ENVVAR_REPO_DIR.joinpath(".cache")  # used in `.env`

    ENVVAR_addon_path: Path = ENVVAR_REPO_DIR.joinpath(
        "mosplat_blender"
    )  # used in `.env` and argparse

    defaults = ArgparseDefaults(addon_src_dir=ENVVAR_addon_path)

    args = get_args(defaults)  # get program args from argparse

    if args.addon_src_dir != ENVVAR_addon_path:
        print(
            f"Overriding default addon source dir: \n{ENVVAR_addon_path=}\n{args.addon_src_dir=}"
        )
        ENVVAR_addon_path = args.addon_src_dir

    timestamp: datetime.datetime = datetime.datetime.now()
    timestamp_str = f"# auto-generated in build: {timestamp} \n"

    wheels_dir: Path = Path(os.path.join(args.addon_src_dir, "wheels"))
    print(f"Wheels Directory Path: {wheels_dir}")

    REQUIREMENTS_TXT_FILE: Path = Path(
        os.path.join(args.addon_src_dir, "requirements.txt")
    )
    print(f"`requirements.txt` File Path: {REQUIREMENTS_TXT_FILE}")

    MANIFEST_TXT_FILE: Path = Path(
        os.path.join(args.addon_src_dir, "blender_manifest.txt")
    )
    print(f"Input `blender_manifest.txt` File Path: {MANIFEST_TXT_FILE}")
    manifest_toml_file: Path = Path(
        os.path.join(args.addon_src_dir, "blender_manifest.toml")
    )
    print(f"Output `blender_manifest.toml` File Path: {manifest_toml_file}")

    DOTENV_TXT_FILE: Path = Path(os.path.join(args.addon_src_dir, ".env.txt"))
    print(f"Input `.env.txt` File Path: {DOTENV_TXT_FILE}")
    dotenv_file: Path = Path(os.path.join(args.addon_src_dir, ".env"))
    print(f"Output `.env` File Path: {dotenv_file}")

    return (
        BuildContext(
            timestamp_str,
            wheels_dir,
            REQUIREMENTS_TXT_FILE,
            MANIFEST_TXT_FILE,
            manifest_toml_file,
            DOTENV_TXT_FILE,
            dotenv_file,
            ENVVAR_REPO_DIR,
            ENVVAR_addon_path,
            envvar_cache_dir,
        ),
        args,
    )


def install_pypi_wheels(requirements_txt_file):
    """install wheels for development"""
    print("Beginning install...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_txt_file)]
        )

    except subprocess.CalledProcessError as e:
        print(f"Error while installing PyPI wheels for development: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("Install complete.")


def download_pypi_wheels(wheels_dir, requirements_txt_file, version, install):
    """use `subprocess` to invoke `pip download ...` on the addon's `requirements.txt`."""
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "-r",
                str(requirements_txt_file),
                "--dest",
                str(wheels_dir),
                "--only-binary=:all:",
                f"--python-version={version}",
            ]
        )

    except subprocess.CalledProcessError as e:
        print(f"Error in call to download PyPI wheels: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("All PyPI wheels successfully downloaded.")

    if install:
        print()  # skip a line
        install_pypi_wheels(requirements_txt_file)


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

    with open(ctx.MANIFEST_TXT_FILE, "r", encoding="utf-8") as f:
        next(f)  # skip the comment on the first line
        template = f.read()

    with open(ctx.manifest_toml_file, "w", encoding="utf-8") as f:
        f.write(ctx.timestamp_str)
        f.write(template.format(wheels_block=WHEELS_STR))

    print(f"`{ctx.manifest_toml_file}` successfully generated.")


def generate_dotenv(ctx: BuildContext):
    """
    format the variables within the template `.env.txt` & save as `.env`.
    """

    print(f"`Generating `.env`...")

    with open(ctx.DOTENV_TXT_FILE, "r", encoding="utf-8") as f:
        template = f.read()

    template = template.format(
        repo_dir=ctx.ENVVAR_REPO_DIR.as_posix(),
        addon_src_dir=ctx.ENVVAR_ADDON_SRC_DIR.as_posix(),
        cache_dir=ctx.envvar_cache_dir.as_posix(),
    )

    with open(ctx.dotenv_file, "w", encoding="utf-8") as f:
        f.write(ctx.timestamp_str)
        f.write(template)

    print(f"`{ctx.dotenv_file}` successfully generated.")


def clean(wheels_dir):
    print(f"Beginning clean...")
    if os.path.exists(wheels_dir) and os.path.isdir(wheels_dir):
        try:
            shutil.rmtree(wheels_dir)
            print(f"Wheels directory and all its contents have been removed.")
        except OSError as e:
            print(f"Error deleting wheels directory: : {e}")

    print(f"Clean complete.")


def main():
    ctx, args = prepare_context()

    for p in [
        ctx.REQUIREMENTS_TXT_FILE,
        ctx.MANIFEST_TXT_FILE,
        ctx.DOTENV_TXT_FILE,
        ctx.ENVVAR_REPO_DIR,
    ]:  # check that all required input resources are where they are supposed to be
        if not os.path.exists(ctx.wheels_dir):
            raise RuntimeError(f"Expected {p!r} in file system, but was not found.")

    print()  # skip a line

    if args.clean:
        clean(ctx.wheels_dir)
        print()  # skip a line

    download_pypi_wheels(
        ctx.wheels_dir,
        ctx.REQUIREMENTS_TXT_FILE,
        args.version,
        install=args.dev,
    )
    print()  # skip a line

    generate_blender_manifest_toml(ctx)
    print()  # skip a line

    generate_dotenv(ctx)


if __name__ == "__main__":
    main()
