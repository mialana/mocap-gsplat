"""decorator functions"""

from __future__ import annotations

from functools import wraps, update_wrapper, partial
from typing import Callable, ParamSpec, TypeVar, Type, Optional
from .schemas import DeveloperError
from .constants import _MISSING_

# maintains original callable's signature for `run_once` and `record_work_time`
P = ParamSpec("P")
# maintains orig callable's returntype  for `run_once`  and `record_work_time`
R = TypeVar("R")
# tracks decorated class for `no_instantiate`
T = TypeVar("T", bound=Type)


def run_once(f: Callable[P, R]) -> Callable[P, R]:
    """
    executes the decorated function at most once.
    it takes advantage of the fact that Python functions are just objects.
    first call dynamically stores the return value as an attribute on the function itself,
    as well as a boolean value to track that it has ran.
    then, subsequent calls can return the cached result without re-executing the function.
    """

    @wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if not getattr(wrapper, "_has_run", False):
            setattr(wrapper, "_result", f(*args, **kwargs))
            setattr(wrapper, "_has_run", True)
        return getattr(wrapper, "_result")

    return wrapper


class run_once_per_instance(object):
    """decorator class that runs a method only once per class instance."""

    def __init__(self, fn):
        update_wrapper(self, fn)
        self.fn = fn

        self._has_run_attr = f"__{fn.__name__}_has_run"
        self._run_result_attr = f"__{fn.__name__}_run_result"

    def __call__(self, instance, *args, **kwargs):
        has_run = getattr(instance, self._has_run_attr, False)
        result = getattr(instance, self._run_result_attr, _MISSING_)
        if not has_run or result is _MISSING_:
            result = self.fn(instance, *args, **kwargs)
            setattr(instance, self._has_run_attr, True)
            setattr(instance, self._run_result_attr, result)
        return result

    def __get__(self, instance, instancetype):
        if instance is None:
            return self

        def bound(*args, **kwargs):
            return self(instance, *args, **kwargs)

        update_wrapper(bound, self.fn)
        return bound


def no_instantiate(cls: T) -> T:
    """
    a decorater for classes that ensure they are never instantiated.
    i.e. they are classes for namespacing-purposes (e.g. `interfaces.MosplatLoggingInterface`)
    """

    def __new__(cls_, *args, **kwargs):
        raise DeveloperError(f"{cls_.__name__} cannot be instantiated")

    cls.__new__ = staticmethod(__new__)
    return cls


def record_work_time(
    fn: Callable[P, R],
    start_callback: Optional[Callable[[float], None]] = None,
    end_callback: Optional[Callable[[float, float], None]] = None,
) -> Callable[P, R]:
    """
    a decorator / function wrapper that will track execution time
    and execute given callbacks.
    `start_time` and `end_time` are float values determined using `datetime.now().timestamp()

    Args:
        fn: Any function, whose exact signature will be preserved.
        start_callback:
            def start_callback(start_time: float) -> None: ...
        end_callback:
            def end_callback(end_time: float, time_elapsed: float) -> None: ...

    Returns:
        Wrapped function (with exact signature).
    """
    from datetime import datetime
    import time

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start_time = datetime.now().timestamp()
        start_counter = time.perf_counter()

        if start_callback is not None:
            start_callback(start_time)

        ret = fn(*args, **kwargs)

        end_time = datetime.now().timestamp()
        end_counter = time.perf_counter()
        time_elapsed = end_counter - start_counter

        if end_callback is not None:
            end_callback(end_time, time_elapsed)

        return ret

    return wrapper
