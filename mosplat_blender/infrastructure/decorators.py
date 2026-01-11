from functools import wraps
from typing import Callable, cast

from .protocols import R, SupportsRunOnce


def run_once(f: Callable[..., R]) -> SupportsRunOnce[R]:
    def impl(*args, **kwargs):
        if not wrapped.has_run:
            wrapped.result = f(*args, **kwargs)
            wrapped.has_run = True
        return wrapped.result

    wrapper = wraps(f)(impl)
    wrapped = cast(SupportsRunOnce[R], wrapper)
    wrapped.has_run = False
    wrapped.result = None
    return wrapped