import os
import subprocess
import sys
from pathlib import Path

ADDON_HUMAN_READABLE = os.getenv("ADDON_HUMAN_READABLE", "mosplat_blender")


def inplace(target_directories: list[Path]):
    print("Starting format in-place.")
    try:
        subprocess.check_call([sys.executable, "-m", "isort", *target_directories])
        print("isort format in-place succeeded.")
    except subprocess.CalledProcessError as e:
        e.add_note("isort format in-place failed.")
        raise

    try:
        subprocess.check_call([sys.executable, "-m", "black", *target_directories])
        print("black succeeded.")
    except subprocess.CalledProcessError as e:
        e.add_note("black format in-place failed.")
        raise

    print("Done.")


def check(target_directories: list[Path]):
    print("Starting format check.")

    isort_proc = subprocess.run(
        [sys.executable, "-m", "isort", "--check-only", "--diff", *target_directories],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    isort_result = isort_proc.returncode
    print(f"isort check returncode: '{isort_result}'")

    black_proc = subprocess.run(
        [sys.executable, "-m", "black", "--check", "--diff", *target_directories],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    black_result = black_proc.returncode
    print(f"black check returncode: '{black_result}'")

    print("Done.")

    if isort_result or black_result:
        raise SystemExit("Format check failed.")


def main():
    ADDON_SRC_DIR = Path(__file__).resolve().parents[1] / ADDON_HUMAN_READABLE
    SCRIPTS_DIR = Path(__file__).resolve().parent

    target_directories = [ADDON_SRC_DIR, SCRIPTS_DIR]

    is_check = len(sys.argv) > 1 and sys.argv[1] == "--check"

    if is_check:
        check(target_directories)
    else:
        inplace(target_directories)


if __name__ == "__main__":
    main()
