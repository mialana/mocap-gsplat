from pathlib import Path
from statistics import median
from typing import Iterable, TypeVar, List, Tuple, TypeGuard, Type


def int_median(iter: Iterable[int]) -> int:
    return int(round(median(iter))) if iter else -1


T = TypeVar("T")


def append_if_not_equals(iter: List[T], *, item: T, target: T) -> None:
    if item != target:
        iter.append(item)


def is_path_accessible(p: Path) -> bool:
    if not p.exists():
        return False
    try:
        p.stat()
        return True
    except (PermissionError, OSError):
        return False


def tuple_matches_type_tuple(value_tuple: Tuple, type_tuple: Tuple) -> bool:
    if len(value_tuple) != len(type_tuple):
        return False
    return all(isinstance(v, t) for v, t in zip(value_tuple, type_tuple))
