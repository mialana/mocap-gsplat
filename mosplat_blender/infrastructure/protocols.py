"""
protocol classes for static-typing purposes.
many exist to prevent escape of blender types into .infrastructure
"""

from dataclasses import Field
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterator,
    Protocol,
    Self,
    TypeVar,
    runtime_checkable,
)

T = TypeVar("T", covariant=True)


@runtime_checkable
class SupportsCollectionProperty(Protocol[T]):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[T]: ...
    def __getitem__(self, index: int) -> T: ...

    def add(self) -> T: ...
    def remove(self, index: int) -> None: ...
    def clear(self) -> None: ...


@runtime_checkable
class SupportsDataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


@runtime_checkable
class SupportsToFromDict(Protocol):
    def to_dict(self) -> Dict: ...

    @classmethod
    def from_dict(cls, d: Dict) -> Self: ...
