#!/usr/bin/env python3
"""
Author : Amy Liu <aliu@amyliu.dev>
Purpose: Download PyPI wheels & generate `blender_manifest.toml` for Mo-splat Blender add-on.
"""

import argparse
from pathlib import Path
import sys
import os
import subprocess


def get_args():
    parser = argparse.ArgumentParser(
        description="Download PyPI wheels & generate `blender_manifest.toml` for Mo-splat Blender add-on.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    current_dir: Path = os.path.dirname(os.path.abspath(__file__))
    default_addon_path: Path = os.path.abspath(
        os.path.join(current_dir, "..", "mosplat_blender")
    )
    parser.add_argument(
        "-p",
        "--path",
        help="Path to Mo-splat Blender add-on directory",
        type=Path,
        default=default_addon_path,
    )

    parser.add_argument(
        "-v",
        "--version",
        help="Version of Python interpreter within current Blender installation",
        type=str,
        default="3.11",
    )

    return parser.parse_args()


def download_pypi_wheels(addon_dir, wheels_dir, version):
    """Use `subprocess` to invoke `pip download ...` on the addon's `requirements.txt`."""
    requirements_txt_file: Path = os.path.join(addon_dir, "requirements.txt")
    print(f"`requirements.txt` File Path: {requirements_txt_file}")

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

    print("All PyPI wheels successfully installed.")


def generate_blender_manifest_toml(addon_dir, wheels_dir):
    """
    Append the downloaded `*.whl` files to the template `blender_manifest.txt file
    & save as `blender_manifest.toml`.
    """

    wheels = sorted(
        f'  "./wheels/{f}"' for f in os.listdir(wheels_dir) if f.endswith(".whl")
    )

    manifest_txt_file: Path = os.path.join(addon_dir, "blender_manifest.txt")
    print(f"`blender_manifest.txt` File Path: {manifest_txt_file}")

    manifest_toml_file: Path = os.path.join(addon_dir, "blender_manifest.toml")
    print(f"`blender_manifest.toml` File Path: {manifest_toml_file}")

    with open(manifest_txt_file, "r", encoding="utf-8") as f:
        next(f)  # skip the comment on the first line
        template = f.read()

    with open(manifest_toml_file, "w", encoding="utf-8") as f:
        f.write(
            template.format(wheels_block="wheels = [\n" + ",\n".join(wheels) + "\n]\n")
        )

    print(f"`{manifest_toml_file}` successfully generated.")


def main():
    args = get_args()
    print(f"Addon Path: {args.path}")
    print(f"Blender Python verion: {args.version}")

    wheels_dir: Path = os.path.join(args.path, "wheels")
    print(f"Wheels Directory Path: {wheels_dir}")

    download_pypi_wheels(args.path, wheels_dir, args.version)
    print("")  # skip a line
    generate_blender_manifest_toml(args.path, wheels_dir)


if __name__ == "__main__":
    main()
