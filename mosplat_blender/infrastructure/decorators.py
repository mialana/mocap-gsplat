"""decorator functions"""

from bpy.types import Context, Event  # import `bpy` to enforce types

from functools import wraps, partial
from typing import (
    Callable,
    ParamSpec,
    TypeVar,
    Type,
    Concatenate,
    TYPE_CHECKING,
    Any,
    TypeAlias,
    TypedDict,
    Required,
    NotRequired,
    Unpack,
    Union,
)

from queue import Queue
from threading import Event as ThreadingEvent

from ..interfaces import MosplatWorkerInterface

if TYPE_CHECKING:
    from ..core.operators import MosplatOperatorBase
else:
    MosplatOperatorBase: TypeAlias = Any


P = ParamSpec(
    "P"
)  # maintains original callable's signature for `run_once` and `worker_fn`
R = TypeVar("R")  # maintains orig callable's returntype  for `run_once`
T = TypeVar("T", bound=Type)  # tracks decorated class for `no_instantiate`

OpT = TypeVar(
    "OpT", bound=MosplatOperatorBase
)  # types the `self` parameter for `worker_fn_auto` and `encapsulated_context`


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
        raise RuntimeError(f"{cls_.__name__} cannot be instantiated")

    cls.__new__ = staticmethod(__new__)
    return cls


def worker_fn_auto(
    fn: Callable[Concatenate[OpT, Queue, ThreadingEvent, P], None],
) -> Callable[Concatenate[OpT, P], None]:
    """a decorator that creates a closure of the worker creation and other abstractable setup"""

    def wrapper(
        self: OpT,
        *args: P.args,
        **kwargs: P.kwargs,
    ):

        self._worker = MosplatWorkerInterface(worker_fn=partial(fn, *args, **kwargs))
        self._worker.start()

        self._timer = self._wm.event_timer_add(
            time_step=0.1, window=self._context.window
        )
        self._wm.modal_handler_add(self)

    return wrapper


class WithContextKwargs(TypedDict):
    context: Required[Context]
    event: NotRequired[Event]


def encapsulated_context(
    fn: Callable[Concatenate[OpT, Context, ...], R],
) -> Callable[Concatenate[OpT, Context, ...], R]:
    """a decorator that sets `_context` properties before function runs, and removes it when complete."""

    def wrapper(self: OpT, *args, **kwargs: Unpack[WithContextKwargs]):

        context = kwargs["context"]
        self._context = context
        self._props._context = context
        self._prefs._context = context

        try:
            return fn(self, context, *args, **kwargs)
        finally:
            self._context = None
            self._props._context = None
            self._prefs._context = None

    return wrapper
