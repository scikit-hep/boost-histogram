from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Tuple, Union

if TYPE_CHECKING:
    from builtins import ellipsis

    from numpy import ufunc as Ufunc
    from numpy.typing import ArrayLike

    from boost_histogram._core.accumulators import Mean, WeightedMean, WeightedSum
    from boost_histogram._core.hist import _BaseHistogram as CppHistogram

    Accumulator = Union[WeightedSum, Mean, WeightedMean]
else:
    ArrayLike = Any
    Ufunc = Any
    CppHistogram = Any
    Accumulator = Any


__all__ = (
    "CppHistogram",
    "AxisLike",
    "ArrayLike",
    "Ufunc",
    "StdIndex",
    "StrIndex",
)


class AxisLike(Protocol):
    def index(self, value: float | str) -> int: ...

    def __len__(self) -> int: ...


StdIndex = Union[int, slice, "ellipsis", Tuple[Union[slice, int, "ellipsis"], ...]]
StrIndex = Union[
    int, slice, str, "ellipsis", Tuple[Union[slice, int, str, "ellipsis"], ...]
]
