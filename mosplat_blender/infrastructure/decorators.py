"""decorator functions"""

from functools import wraps
from typing import Callable

from .constants import P, R


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
