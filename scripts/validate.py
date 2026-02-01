from pathlib import Path
import os

ADDON_HUMAN_READABLE = os.getenv("ADDON_HUMAN_READABLE", "mosplat_blender")


def main():
    zip_path = Path(__file__).resolve().parent.parent / f"{ADDON_HUMAN_READABLE}.zip"

    if not zip_path.exists():
        print(f"Zip created from build does not exist at path on system: {zip_path}")
        return

    max_size = 2 * 1024 * 1024 * 1024  # 2GiB

    size = zip_path.stat().st_size
    print(f"Zip size: {size / (1024**2):.2f} MB")

    if size > max_size:
        raise SystemExit("Zip exceeds GitHub Releases 2GB limit")


if __name__ == "__main__":
    main()
