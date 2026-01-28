from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Iterable, TypeVar, List, Tuple, TYPE_CHECKING
import sys

if TYPE_CHECKING:
    import cv2


def int_median(iter: Iterable[int]) -> int:
    return int(round(median(iter))) if iter else -1


T = TypeVar("T")


def append_if_not_equals(iter: List[T], *, item: T, target: T) -> None:
    if item != target:
        iter.append(item)


def try_access_path(p: Path):
    """raises `UserFacingError` on failure"""

    from .schemas import UserFacingError  # keep import contained

    if not p.exists():
        raise UserFacingError(f"'{p}' does not exist.")
    try:
        p.stat()
        return
    except (PermissionError, OSError) as e:
        raise UserFacingError(f"'{p}' is not accessible.") from e


def is_path_accessible(p: Path) -> bool:
    from .schemas import UserFacingError  # keep import contained

    try:
        try_access_path(p)
        return True
    except UserFacingError:
        return False


def tuple_matches_type_tuple(value_tuple: Tuple, type_tuple: Tuple) -> bool:
    if len(value_tuple) != len(type_tuple):
        return False
    return all(isinstance(v, t) for v, t in zip(value_tuple, type_tuple))


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
            pass


def write_frame_data_to_npy(
    frame_idx: int, caps: List[cv2.VideoCapture], out_path: Path
):
    import cv2
    import numpy as np

    from .schemas import UserFacingError  # keep import contained

    images: List[cv2.typing.MatLike] = []
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            raise UserFacingError(f"Failed to read frame: {frame_idx}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
        images.append(frame)

    stacked = np.stack(images, axis=0)
    np.save(out_path, stacked)
