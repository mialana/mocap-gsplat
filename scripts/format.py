from pathlib import Path
import sys
import os
import subprocess

ADDON_HUMAN_READABLE = os.getenv("ADDON_HUMAN_READABLE", "mosplat_blender")


def main(addon_src_dir: Path):
    try:
        subprocess.check_call([sys.executable, "-m", "isort", str(addon_src_dir)])
        print("isort succeeded.")
    except subprocess.CalledProcessError as e:
        print("isort failed.")
        raise

    try:
        subprocess.check_call([sys.executable, "-m", "black", str(addon_src_dir)])
        print("black succeeded.")
    except subprocess.CalledProcessError as e:
        print("black failed.")
        raise

    print("Done.")


if __name__ == "__main__":
    addon_src_dir = Path(__file__).resolve().parent.parent / ADDON_HUMAN_READABLE
    main(addon_src_dir)
