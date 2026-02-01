"""generates meta for all files in `core` module that contain blender property types."""

import ast
from pathlib import Path
import argparse
import datetime


BLENDER_PROPERTY_TYPES = {
    "BoolProperty",
    "IntProperty",
    "FloatProperty",
    "StringProperty",
    "EnumProperty",
    "PointerProperty",
    "CollectionProperty",
    "IntVectorProperty",
}

timestamp: datetime.datetime = datetime.datetime.now()
timestamp_str = f"# auto-generated in build: {timestamp} \n\n"


class PropertyGroupMetaExtractor(ast.NodeVisitor):
    def __init__(self):
        self.classes: dict[str, dict[str, dict[str, str]]] = {}

    def visit_ClassDef(self, node: ast.ClassDef):
        props: dict[str, dict[str, str]] = {}

        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign):
                continue

            if not isinstance(stmt.annotation, ast.Call):
                continue

            call = stmt.annotation

            def get_call_name(func: ast.expr) -> str | None:
                if isinstance(func, ast.Name):
                    return func.id
                if isinstance(func, ast.Attribute):
                    return func.attr
                return None

            call_name = get_call_name(call.func)
            if call_name not in BLENDER_PROPERTY_TYPES:
                continue

            if not isinstance(stmt.target, ast.Name):
                continue

            prop_name = stmt.target.id

            name: str = ""
            description: str = ""

            for kw in call.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    name = str(kw.value.value)
                elif kw.arg == "description" and isinstance(kw.value, ast.Constant):
                    description = str(kw.value.value)

            props.setdefault(
                prop_name,
                {
                    "name": name or "",
                    "description": description or "",
                },
            )

        if props:
            self.classes[node.name] = props


def generate_meta_file(py_file: Path):
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source)

    extractor = PropertyGroupMetaExtractor()
    extractor.visit(tree)

    if not extractor.classes:
        return  # nothing to do

    meta_lines = [
        timestamp_str,
        "from __future__ import annotations",
        "",
    ]

    for cls_name, props in extractor.classes.items():
        meta_lines.append(f"{cls_name}Meta = {{")
        for prop, data in props.items():
            meta_lines.append(
                f"    {prop!r}: {{'name': {data['name']!r}, 'description': {data['description']!r}}},"
            )
        meta_lines.append("}")
        meta_lines.append("")

    meta_path = py_file.with_name(py_file.stem + "_meta.py")

    with meta_path.open("w", encoding="utf-8", newline="\n") as file:
        file.writelines("\n".join(meta_lines))

    print(f"Generated '{meta_path}'")


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-a",
        "--addon_src_dir",
        help="Path to add-on source directory",
        type=Path,
    )

    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        type=Path,
        help="Files that need processing",
        default=[],
    )

    return parser.parse_args()


def main(addon_src_dir: Path):
    CORE_MODULE = addon_src_dir / "core"

    py_files = [file for file in CORE_MODULE.rglob("*.py")]

    py_files.extend(args.files)

    for file in py_files:
        generate_meta_file(file)


if __name__ == "__main__":
    args = get_args()

    ADDON_SRC_DIR: Path = (
        args.addon_src_dir or Path(__file__).resolve().parent.parent / "mosplat_blender"
    )

    main(addon_src_dir=ADDON_SRC_DIR)
