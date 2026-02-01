import argparse
import os
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
}

timestamp: datetime = datetime.now()
timestamp_str = f"# {timestamp} \n# created using '{Path(__file__).name}'\n"


check_only: bool = False
verbose: bool = True
found_file_count: int = 0
modification_count: int = 0  # tracks how many files need modification / are modified


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
        self.symbols = set(symbols)
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

        import_node = cst.ImportFrom(
            module=self._from_dotted_name(self.module_name),
            names=[cst.ImportAlias(cst.Name(s)) for s in sorted(self.symbols)],
            relative=[cst.Dot()],
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
        parts = name.split(".")
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
            if isinstance(stmt, cst.FunctionDef) and stmt.name.value == "_meta":
                continue
            new_body.append(stmt)

        new_body.append(self._make_meta_property(original_node.name.value))
        return updated_node.with_changes(
            body=updated_node.body.with_changes(body=new_body)
        )

    def _make_meta_property(self, cls_name: str) -> cst.FunctionDef:
        return cst.FunctionDef(
            name=cst.Name("_meta"),
            params=cst.Parameters(params=[cst.Param(cst.Name("self"))]),
            decorators=[cst.Decorator(cst.Name("property"))],
            body=cst.IndentedBlock(
                body=[
                    cst.SimpleStatementLine(
                        body=[cst.Return(cst.Name(f"{cls_name}_Meta"))]
                    )
                ]
            ),
        )


def patch_original_file(py_file: Path, meta_path: Path, meta_symbols: list[str]):
    source = py_file.read_text(encoding="utf-8")
    module = cst.parse_module(source)

    rel_path = meta_path.relative_to(py_file.parent)
    parts = rel_path.with_suffix("").parts
    module_name = ".".join(parts)

    module = module.visit(MetaImportInjector(module_name, meta_symbols))

    module = module.visit(
        MetaPropertyInjector(classes_with_meta={name[:-4] for name in meta_symbols})
    )

    new_source = module.code
    sorted = isort.code(new_source)  # sort with isort

    # format with black
    formatted = black.format_str(sorted, mode=black.FileMode())

    needs_modification = formatted != source

    print(f"({found_file_count}+) Analyzed original file patch status.")
    if verbose:
        print(f"\tCurrent character count: '{len(source)}'")
        print(f"\tPatched character count: '{len(formatted)}'")
        print(f"\tNeeds modification: '{needs_modification}'")

    global modification_count
    modification_count += int(needs_modification)

    if check_only and (verbose or needs_modification):
        print(
            f"Original File Check Result: {('failed' if needs_modification else 'success').upper()}"
        )
    elif needs_modification:  # commit
        py_file.write_text(formatted, encoding="utf-8", newline="\n")
        print(f"PATCHED '{py_file}'.")


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

    global found_file_count
    found_file_count += 1

    meta_lines = [
        timestamp_str,
    ]

    prop_count = 0
    for cls_name, props in extractor.classes.items():
        meta_lines.append(f"{cls_name}_Meta: dict[str, dict[str, str]] = {{")
        for prop, data in props.items():
            meta_lines.append(
                f"    {prop!r}: {{'name': {data['name']!r}, 'description': {data['description']!r}}},"
            )
            prop_count += 1
        meta_lines.append("}")
        meta_lines.append("")

    meta_dir = py_file.parent / "meta"
    meta_dir.mkdir(exist_ok=True)

    meta_path = meta_dir / f"{py_file.stem}_meta.py"

    meta_exists = meta_path.exists()

    meta_code = "\n".join(meta_lines)
    formatted = black.format_str(meta_code, mode=black.FileMode())  # format with black

    needs_modification = diff_meta_file(meta_path, formatted)

    global modification_count
    modification_count += int(needs_modification)

    print(
        f"({found_file_count}) Found file containing Blender properties: '{py_file.name}'"
    )
    if verbose:
        print(
            f"\tClasses found containing Blender properties: '{len(extractor.classes)}'"
        )
        print(f"\tTotal Blender properties in file: '{prop_count}'")
        print(f"\tHas meta file been generated previously: '{meta_exists}'")
        print(f"\tNeeds modification: '{needs_modification}'")

    if check_only and (verbose or needs_modification):
        print(
            f"Meta File Check Result: {('Failed' if needs_modification else 'Success').upper()}"
        )
    elif needs_modification:  # commit
        meta_path.write_text(formatted, encoding="utf-8", newline="\n")
        print(f"GENERATED '{meta_path}'")

    meta_symbols = [f"{cls}Meta" for cls in extractor.classes]
    patch_original_file(py_file, meta_path, meta_symbols)


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
    global check_only, verbose
    check_only = check
    verbose = not quiet

    CORE_MODULE = addon_src_dir / "core"

    py_files = [file for file in CORE_MODULE.rglob("*.py")]

    py_files.extend(extra_files)

    for file in py_files:
        generate_meta_file(file)

    print("Done.")

    if check_only and modification_count > 0:
        raise SystemExit(
            f"SYSTEM ERROR: '{modification_count}' files still need changes applied."
        )

    print(
        f"{'Files needing modification' if check_only else 'Files modified'}: '{modification_count}'"
    )


if __name__ == "__main__":
    args = get_args()

    ADDON_SRC_DIR: Path = (
        args.addon_src_dir
        or Path(__file__).resolve().parent.parent / ADDON_HUMAN_READABLE
    )

    main(
        addon_src_dir=ADDON_SRC_DIR,
        extra_files=args.files,
        check=args.check,
        quiet=args.quiet,
    )
