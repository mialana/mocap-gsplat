import argparse
import os
import re
from datetime import datetime
from pathlib import Path

import black
import isort
import libcst as cst
from libcst import matchers as m

ADDON_HUMAN_READABLE = os.getenv("ADDON_HUMAN_READABLE", "mosplat_blender")

BLENDER_PROPERTY_TYPES = {
    "BoolProperty",
    "IntProperty",
    "FloatProperty",
    "StringProperty",
    "EnumProperty",
    "PointerProperty",
    "CollectionProperty",
    "IntVectorProperty",
    "FloatVectorProperty",
}

pyproject_toml_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
ISORT_CONFIG = isort.Config(settings_file=str(pyproject_toml_path))

timestamp: datetime = datetime.now()
TIMESTAMP_STR = f"# {timestamp} \n# created using '{Path(__file__).name}'\n"


CHECK_ONLY: bool = False
VERBOSE: bool = True
FOUND_FILE_COUNT: int = 0
MODIFICATION_COUNT: int = 0  # tracks how many files need modification / are modified
SCHEMAS_MODULE: Path = Path()
ADDON_SRC_DIR: Path = Path()


class PropertyGroupMetaExtractor(cst.CSTVisitor):
    def __init__(self):
        self.classes: dict[str, dict[str, dict[str, str]]] = {}
        self._current_class: str | None = None

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self._current_class = node.name.value
        self.classes.setdefault(self._current_class, {})

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        if not self._current_class:
            return
        if not self.classes[self._current_class]:
            self.classes.pop(self._current_class, None)
        self._current_class = None

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        if self._current_class is None:
            return

        annotation = node.annotation.annotation
        if not isinstance(annotation, cst.Call):
            return

        func = annotation.func
        if isinstance(func, cst.Name):
            call_name = func.value
        elif isinstance(func, cst.Attribute):
            call_name = func.attr.value
        else:
            return

        if call_name not in BLENDER_PROPERTY_TYPES:
            return

        if not isinstance(node.target, cst.Name):
            return

        prop_name = node.target.value
        name: str = ""
        description: str = ""

        for arg in annotation.args:
            if arg.keyword is None:
                continue

            if arg.keyword.value == "name" and isinstance(arg.value, cst.SimpleString):
                name = str(arg.value.evaluated_value)
            elif arg.keyword.value == "description" and isinstance(
                arg.value, cst.SimpleString
            ):
                description = str(arg.value.evaluated_value)

        self.classes[self._current_class][prop_name] = {
            "name": name,
            "description": description,
        }


class MetaImportInjector(cst.CSTTransformer):
    def __init__(self, module_name: str, symbols: list[str]):
        self.module_name = module_name
        self.symbols = set(symbols + [sym.upper() for sym in symbols])
        self.found = False

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom:
        from collections.abc import Iterable

        if not original_node.module:
            return updated_node

        module = self._to_dotted_name(original_node.module)
        if (
            module
            and module == self.module_name
            and isinstance(original_node.names, Iterable)
            and isinstance(updated_node.names, Iterable)
        ):
            existing = {
                str(name.name.value)
                for name in original_node.names
                if isinstance(name, cst.ImportAlias)
            }

            missing = self.symbols - existing
            if not missing:
                self.found = True
                return updated_node

            new_names = list(updated_node.names)

            for sym in sorted(missing):
                new_names.append(cst.ImportAlias(cst.Name(sym)))

            self.found = True
            return updated_node.with_changes(names=new_names)

        return updated_node

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        if self.found:
            return updated_node

        match = re.search(r"[A-Za-z0-9]", self.module_name)
        assert match
        dot_count = match.start()

        relative = tuple([cst.Dot() for _ in range(dot_count)])

        import_node = cst.ImportFrom(
            module=self._from_dotted_name(self.module_name),
            names=[cst.ImportAlias(cst.Name(s)) for s in sorted(self.symbols)],
            relative=relative,
        )

        body: list[cst.CSTNode] = list(updated_node.body)

        insert_at = 0
        while (
            insert_at < len(body)
            and isinstance(body[insert_at], cst.SimpleStatementLine)
            and (
                m.matches(body[insert_at], m.SimpleStatementLine(body=[m.Expr()]))
                or m.matches(
                    body[insert_at],
                    m.SimpleStatementLine(
                        body=[m.ImportFrom(module=m.Name("__future__"))]
                    ),
                )
            )
        ):
            insert_at += 1

        body.insert(insert_at, import_node)
        body.insert(insert_at + 1, cst.EmptyLine())

        return updated_node.with_changes(body=body)

    @staticmethod
    def _from_dotted_name(name: str) -> cst.Name | cst.Attribute:
        parts = [p for p in name.split(".") if p]
        expr: cst.Name | cst.Attribute = cst.Name(parts[0])
        for part in parts[1:]:
            expr = cst.Attribute(
                value=expr,
                attr=cst.Name(part),
            )
        return expr

    @staticmethod
    def _to_dotted_name(module: cst.Attribute | cst.Name | None) -> str | None:
        if module is None:
            return None
        if isinstance(module, cst.Name):
            return module.value
        if isinstance(module, cst.Attribute):
            parts = []

            base: cst.Attribute | cst.Name = module
            while isinstance(base, cst.Attribute):
                parts.append(base.attr.value)
                if not isinstance(base.value, (cst.Attribute, cst.Name)):
                    break
                base = base.value

            if isinstance(base, cst.Name):
                parts.append(base.value)

            return ".".join(reversed(parts))


class MetaPropertyInjector(cst.CSTTransformer):
    def __init__(self, classes_with_meta: set[str]):
        self.classes_with_meta = classes_with_meta

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if original_node.name.value not in self.classes_with_meta:
            return updated_node

        new_body = []
        for stmt in updated_node.body.body:
            # remove any previous _meta definition
            if (
                isinstance(stmt, cst.SimpleStatementLine)
                and len(stmt.body) == 1
                and isinstance(stmt.body[0], cst.AnnAssign)
                and isinstance(stmt.body[0].target, cst.Name)
                and stmt.body[0].target.value == "_meta"
            ):
                continue
            new_body.append(stmt)

        insert_at = 0

        # Skip class docstring if present
        if (
            new_body
            and isinstance(new_body[0], cst.SimpleStatementLine)
            and len(new_body[0].body) == 1
            and isinstance(new_body[0].body[0], cst.Expr)
            and isinstance(new_body[0].body[0].value, cst.SimpleString)
        ):
            insert_at = 1

        new_body.insert(
            insert_at,
            cst.SimpleStatementLine(
                body=[self._make_meta_classvar(original_node.name.value)]
            ),
        )

        return updated_node.with_changes(
            body=updated_node.body.with_changes(body=new_body)
        )

    def _make_meta_classvar(self, cls_name: str) -> cst.AnnAssign:
        meta_type = f"{cls_name}_Meta"
        meta_instance = f"{cls_name}_META".upper()
        return cst.AnnAssign(
            target=cst.Name("_meta"),
            annotation=cst.Annotation(cst.Name(meta_type)),
            value=cst.Name(meta_instance),
        )


def patch_original_file(
    py_file: Path, meta_path: Path, meta_symbols: list[str], original_classes: list[str]
):
    source = py_file.read_text(encoding="utf-8")
    module = cst.parse_module(source)

    # module_name = dot_abs_path(meta_path, root_dir=ADDON_SRC_DIR)
    module_name = dot_rel_path(meta_path, source_path=py_file)

    module = module.visit(MetaImportInjector(module_name, meta_symbols))

    module = module.visit(
        MetaPropertyInjector(classes_with_meta={cls for cls in original_classes})
    )

    new_source = module.code

    sorted = isort.code(new_source, config=ISORT_CONFIG)  # sort with isort

    # format with black
    formatted = black.format_str(sorted, mode=black.FileMode())

    needs_modification = formatted != source

    print(f"({FOUND_FILE_COUNT}+) Analyzed original file patch status.")
    if VERBOSE:
        print(f"\tCurrent character count: '{len(source)}'")
        print(f"\tPatched character count: '{len(formatted)}'")
        print(f"\tNeeds modification: '{needs_modification}'")

    global MODIFICATION_COUNT
    MODIFICATION_COUNT += int(needs_modification)

    if CHECK_ONLY and (VERBOSE or needs_modification):
        print(
            f"Original File Check Result: {('failed' if needs_modification else 'success').upper()}"
        )
    elif needs_modification:  # commit
        py_file.write_text(formatted, encoding="utf-8", newline="\n")
        print(f"PATCHED '{py_file}'.")


def dot_rel_path(dest_path: Path, *, source_path: Path) -> str:
    source_path = source_path.resolve()
    dest_path = dest_path.resolve()

    rel = os.path.relpath(dest_path, source_path)

    # drop ".py"
    rel = os.path.splitext(rel)[0]

    parts = rel.split(os.sep)

    dots = ""
    while parts and parts[0] == "..":
        dots += "."
        parts.pop(0)

    # always at least one dot for relative import
    dots = dots or "."

    if parts:
        return dots + ".".join(parts)
    else:
        return dots


def dot_abs_path(dest_path: Path, *, root_dir: Path) -> str:
    dest_path = dest_path.resolve()

    rel = os.path.relpath(dest_path, start=root_dir)

    # drop ".py"
    rel = os.path.splitext(rel)[0]

    parts = [root_dir.name] + rel.split(os.sep)

    if parts:
        return ".".join(parts)
    else:
        return rel


def diff_meta_file(meta_path: Path, formatted_new_text: str) -> bool:
    """returns true if there is a difference"""

    if not meta_path.exists():
        return True

    old_lines = meta_path.read_text(encoding="utf-8").splitlines()
    new_lines = formatted_new_text.split("\n")[:-1]

    return old_lines[1:] != new_lines[1:]  # remove timestamp line from check


def generate_meta_file(py_file: Path):
    source = py_file.read_text(encoding="utf-8")
    module = cst.parse_module(source)

    extractor = PropertyGroupMetaExtractor()
    module.visit(extractor)

    if not extractor.classes:
        return  # doesn't contain blender properties

    global FOUND_FILE_COUNT
    FOUND_FILE_COUNT += 1

    meta_dir = py_file.parent / "meta"
    meta_dir.mkdir(exist_ok=True)

    meta_path = meta_dir / f"{py_file.stem}_meta.py"

    meta_exists = meta_path.exists()

    # import_str = dot_abs_path(SCHEMAS_MODULE, root_dir=ADDON_SRC_DIR)
    import_str = dot_rel_path(SCHEMAS_MODULE, source_path=meta_path)

    meta_lines = [
        TIMESTAMP_STR,
        "",
        f"from {import_str} import PropertyMeta",
        "from typing import NamedTuple",
        "",
    ]

    prop_count = 0
    for cls_name, props in extractor.classes.items():
        meta_lines.append(f"class {cls_name}_Meta(NamedTuple):")
        for prop in props:
            meta_lines.append(f"    {prop}: PropertyMeta")
            prop_count += 1
        meta_lines.append("")

    for cls_name, props in extractor.classes.items():
        meta_lines.append(f"{cls_name.upper()}_META = {cls_name}_Meta(")
        for prop, data in props.items():
            meta_lines.append(
                f"    {prop}=PropertyMeta("
                f"id={prop!r}, name={data['name']!r}, description={data['description']!r}"
                f"),"
            )
        meta_lines.append(")")
        meta_lines.append("")

    meta_code = "\n".join(meta_lines)
    sorted = isort.code(meta_code, config=ISORT_CONFIG)
    formatted = black.format_str(sorted, mode=black.FileMode())  # format with black

    needs_modification = diff_meta_file(meta_path, formatted)

    global MODIFICATION_COUNT
    MODIFICATION_COUNT += int(needs_modification)

    print(
        f"({FOUND_FILE_COUNT}) Found file containing Blender properties: '{py_file.name}'"
    )
    if VERBOSE:
        print(
            f"\tClasses found containing Blender properties: '{len(extractor.classes)}'"
        )
        print(f"\tTotal Blender properties in file: '{prop_count}'")
        print(f"\tHas meta file been generated previously: '{meta_exists}'")
        print(f"\tNeeds modification: '{needs_modification}'")

    if CHECK_ONLY and (VERBOSE or needs_modification):
        print(
            f"Meta File Check Result: {('Failed' if needs_modification else 'Success').upper()}"
        )
    elif needs_modification:  # commit
        meta_path.write_text(formatted, encoding="utf-8", newline="\n")
        print(f"GENERATED '{meta_path}'")

    meta_symbols = [f"{cls}_Meta" for cls in extractor.classes]
    original_classes = [cls for cls in extractor.classes]
    patch_original_file(py_file, meta_path, meta_symbols, original_classes)


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
        nargs="*",
        type=Path,
        help="Files that need processing",
        default=[],
    )

    parser.add_argument(
        "-c",
        "--check",
        action="store_true",
        help="Check only if changes would be made.",
        default=[],
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Output only bare minimum status checks.",
        default=[],
    )

    return parser.parse_args()


def main(
    addon_src_dir: Path,
    extra_files: list[Path] = [],
    check: bool = False,
    quiet: bool = False,
):
    global CHECK_ONLY, VERBOSE, SCHEMAS_MODULE, ADDON_SRC_DIR
    CHECK_ONLY = check
    VERBOSE = not quiet

    ADDON_SRC_DIR = addon_src_dir
    CORE_MODULE = ADDON_SRC_DIR / "core"
    SCHEMAS_MODULE = ADDON_SRC_DIR / "infrastructure" / "schemas.py"

    py_files = [file for file in CORE_MODULE.rglob("*.py")]

    py_files.extend(extra_files)

    for file in py_files:
        generate_meta_file(file)

    print("Done.")

    if CHECK_ONLY and MODIFICATION_COUNT > 0:
        raise SystemExit(f"'{MODIFICATION_COUNT}' files still need changes applied.")

    print(
        f"{'Files needing modification' if CHECK_ONLY else 'Files modified'}: '{MODIFICATION_COUNT}'"
    )


if __name__ == "__main__":
    args = get_args()

    addon_src_dir: Path = (
        args.addon_src_dir or Path(__file__).resolve().parents[1] / ADDON_HUMAN_READABLE
    )

    main(
        addon_src_dir=addon_src_dir,
        extra_files=args.files,
        check=args.check,
        quiet=args.quiet,
    )
