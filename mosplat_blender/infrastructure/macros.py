"""functions here should raise standard library error types."""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_file_location
from os import stat_result
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    List,
    LiteralString,
    Optional,
    Set,
    Tuple,
    TypeAlias,
    TypeGuard,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from infrastructure.schemas import ImagesTensorType

T = TypeVar("T")
K = TypeVar("K", bound=Tuple)


def int_median(iter: Iterable[int]) -> int:
    return int(round(median(iter))) if iter else -1


L = TypeVar("L", bound=LiteralString)
I = TypeVar("I")

Immutable: TypeAlias = Union[Tuple[I, ...], I]

# set is effectively immutable-like for our purposes
ImmutableLike: TypeAlias = Union[Immutable[L], Set[L]]


def immutable_to_set(im: ImmutableLike[L]) -> Set[L]:
    if isinstance(im, set):
        return im

    return set(im) if isinstance(im, tuple) else {im}


def append_if_not_equals(iter: List[T], *, item: T, target: T) -> None:
    if item != target:
        iter.append(item)


def try_access_path(p: Path) -> stat_result:
    """
    will raise `FileNotFoundError`, `OSError`, or `PermissionError`.
    returns the stat result of the file IF it is accessible.
    """

    if not p.exists():
        raise FileNotFoundError(f"'{p}' does not exist.")

    try:
        return p.stat()  # will raise either `OSError` or `PermissionError`
    except (OSError, PermissionError) as e:
        e.add_note(f"NOTE: '{e}' was found but could not retrieve a stat result.")
        raise


def is_path_accessible(p: Path) -> bool:
    try:
        try_access_path(p)
        return True
    except (OSError, PermissionError, FileNotFoundError):
        return False


def tuple_type_matches_known_tuple_type(
    unknown_tuple: Tuple, known_tuple: K
) -> TypeGuard[K]:
    if len(unknown_tuple) != len(known_tuple):
        return False
    return all(type(v) == type(t) for v, t in zip(unknown_tuple, known_tuple))


def kill_subprocess_cross_platform(pid: int):
    if sys.platform != "win32":
        os.killpg(pid, signal.SIGKILL)  # TODO: cross-platform check
    else:
        import psutil

        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except psutil.NoSuchProcess:
            pass  # this requires no exception-raising


def import_module_from_path_dynamic(path: Path) -> ModuleType:
    """raises `ImportError` if a specification could not be loaded from the given path"""
    path = path.resolve()

    spec: Optional[ModuleSpec] = spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from '{path}'")

    module: ModuleType = module_from_spec(spec)
    sys.modules[path.stem] = module  # add to current sys modules
    spec.loader.exec_module(module)  # get the

    return module


def get_required_function(module: ModuleType, name: str) -> Callable:
    try:
        fn = getattr(module, name)
    except AttributeError as e:
        e.add_note(f"NOTE: Module '{module.__name__}' has no function '{name}'")
        raise

    if not isinstance(fn, Callable):
        raise TypeError(f"'{name}' exists but is not callable")

    return fn


def save_tensor_stack_png_preview(tensor: ImagesTensorType, tensor_out_file: Path):
    from torchvision.utils import save_image

    preview_png_file: Path = tensor_out_file.parent / f"{tensor_out_file.stem}.png"

    save_image(tensor, preview_png_file)
