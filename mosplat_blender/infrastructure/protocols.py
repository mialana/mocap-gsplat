from typing import Protocol, TypeVar, Optional

R = TypeVar("R")


class SupportsRunOnce(Protocol[R]):
    has_run: bool
    result: Optional[R]

    def __call__(self, *args, **kwargs) -> R: ...
