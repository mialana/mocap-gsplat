import os
import subprocess
import sys
from pathlib import Path

ADDON_HUMAN_READABLE = os.getenv("ADDON_HUMAN_READABLE", "mosplat_blender")

try:
    terminal_width = os.get_terminal_size().columns
except OSError:
    terminal_width = 80

SEP = "-"


def run_script(call_args: list[str]):
    try:
        subprocess.check_call([sys.executable, *call_args])
    except subprocess.CalledProcessError as e:
        e.add_note(f"Error in prereload script: '{call_args[0]}'")
        raise


def run_script_wrapper(script_name: str, call_args: list[str]):
    start_msg = " '{name}' started.".format(name=script_name)
    print(start_msg.rjust(terminal_width, SEP))

    run_script(call_args)

    end_msg = " finished.".format(name=script_name)
    print(end_msg.rjust(terminal_width))


def main():
    ADDON_SRC_DIR = Path(__file__).resolve().parents[1] / ADDON_HUMAN_READABLE

    SCRIPTS_DIR = Path(__file__).resolve().parent
    FORMAT_SCRIPT = SCRIPTS_DIR / "format.py"
    GENERATE_PROPERTY_META_FILES_SCRIPT = (
        SCRIPTS_DIR / "generate_property_meta_files.py"
    )

    run_script_wrapper(script_name=FORMAT_SCRIPT.name, call_args=[str(FORMAT_SCRIPT)])
    run_script_wrapper(
        script_name=GENERATE_PROPERTY_META_FILES_SCRIPT.name,
        call_args=[
            str(GENERATE_PROPERTY_META_FILES_SCRIPT),
            "-a",
            str(ADDON_SRC_DIR),
            "--quiet",
        ],
    )


if __name__ == "__main__":
    main()
