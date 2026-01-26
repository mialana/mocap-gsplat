"""decorator functions"""

from __future__ import annotations

from functools import wraps, partial
from typing import (
    Callable,
    ParamSpec,
    TypeVar,
    Type,
    Concatenate,
    TYPE_CHECKING,
    TypedDict,
    Required,
    NotRequired,
    Unpack,
    Union,
    TypeAlias,
    Any,
)

from queue import Queue
from threading import Event as ThreadingEvent

from .schemas import DeveloperError

if TYPE_CHECKING:
    from ..core.operators import MosplatOperatorBase
    from ..core.panels import MosplatPanelBase
    from bpy.types import Context, Event  # import `bpy` to enforce types
else:
    MosplatOperatorBase: TypeAlias = Any
    MosplatPanelBase: TypeAlias = Any


P = ParamSpec(
    "P"
)  # maintains original callable's signature for `run_once` and `worker_fn`
R = TypeVar("R")  # maintains orig callable's returntype  for `run_once`
T = TypeVar("T", bound=Type)  # tracks decorated class for `no_instantiate`

OpT = TypeVar(
    "OpT", bound=Union[MosplatOperatorBase, MosplatPanelBase]
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
        raise DeveloperError(f"{cls_.__name__} cannot be instantiated")

    cls.__new__ = staticmethod(__new__)
    return cls


def worker_fn_auto(
    fn: Callable[Concatenate[Queue, ThreadingEvent, P], None],
) -> Callable[Concatenate[MosplatOperatorBase, P], None]:
    """a decorator that creates a closure of the worker creation and other abstractable setup"""

    from ..interfaces import MosplatWorkerInterface  # local import

    def wrapper(
        self: MosplatOperatorBase,
        *args: P.args,
        **kwargs: P.kwargs,
    ):

        self.worker = MosplatWorkerInterface(worker_fn=partial(fn, *args, **kwargs))
        self.worker.start()

        self.timer = self.wm.event_timer_add(time_step=0.1, window=self.context.window)
        self.wm.modal_handler_add(self)

    return wrapper


class WithContextKwargs(TypedDict):
    context: Required[Context]
    event: NotRequired[Event]


def encapsulated_context(
    fn: Callable[Concatenate[OpT, Context, ...], R],
) -> Callable[Concatenate[OpT, Context, ...], R]:
    """a decorator that sets `context` properties before function runs, and removes it when complete."""

    def wrapper(self: OpT, *args, **kwargs: Unpack[WithContextKwargs]):

        context = kwargs["context"]
        self.context = context

        try:
            return fn(self, *args, **kwargs)
        finally:
            self.context = None

    return wrapper
