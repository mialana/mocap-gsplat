"""decorator functions"""

from __future__ import annotations

from functools import wraps
from typing import Callable, ParamSpec, TypeVar, Type
from .schemas import DeveloperError


P = ParamSpec("P")  # maintains original callable's signature for `run_once`
R = TypeVar("R")  # maintains orig callable's returntype  for `run_once`
T = TypeVar("T", bound=Type)  # tracks decorated class for `no_instantiate`


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


def no_instantiate(cls: T) -> T:
    """
    a decorater for classes that ensure they are never instantiated.
    i.e. they are classes for namespacing-purposes (e.g. `interfaces.MosplatLoggingInterface`)
    """

    def __new__(cls_, *args, **kwargs):
        raise DeveloperError(f"{cls_.__name__} cannot be instantiated")

    cls.__new__ = staticmethod(__new__)
    return cls
