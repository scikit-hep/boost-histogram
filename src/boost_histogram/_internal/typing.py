import sys
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Tuple, Type, Union

if sys.version_info < (3, 8):
    from typing_extensions import Protocol, SupportsIndex
else:
    from typing import Protocol, SupportsIndex

if TYPE_CHECKING:
    from numpy import ufunc as Ufunc
    from numpy.typing import ArrayLike
else:
    ArrayLike = Any
    Ufunc = Any


__all__ = (
    "Protocol",
    "SupportsIndex",
    "AxisLike",
    "ArrayLike",
    "Ufunc",
    "StdIndex",
    "StrIndex",
)


class AxisLike(Protocol):
    def index(self, value: float) -> int:
        ...

    def __len__(self) -> int:
        ...


class CppHistogram(Protocol):
    _storage_type: ClassVar[Type[object]]

    def __init__(self, axes: Iterable[Any], storage: Any) -> None:
        ...


StdIndex = Union[int, slice, Tuple[Union[slice, int], ...]]
StrIndex = Union[int, slice, str, Tuple[Union[slice, int, str], ...]]
