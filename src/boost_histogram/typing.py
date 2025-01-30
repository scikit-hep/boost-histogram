from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Union

if TYPE_CHECKING:
    from builtins import ellipsis
    from collections.abc import Sequence

    from numpy import ufunc as Ufunc
    from numpy.typing import ArrayLike
    from uhi.typing.plottable import PlottableAxis

    from boost_histogram._core.accumulators import Mean, WeightedMean, WeightedSum
    from boost_histogram._core.hist import _BaseHistogram as CppHistogram

    Accumulator = Union[WeightedSum, Mean, WeightedMean]
else:
    ArrayLike = Any
    Ufunc = Any
    CppHistogram = Any
    Accumulator = Any


__all__ = (
    "ArrayLike",
    "AxisLike",
    "CppHistogram",
    "RebinProtocol",
    "StdIndex",
    "StrIndex",
    "Ufunc",
)


class AxisLike(Protocol):
    def index(self, value: float | str) -> int: ...

    def __len__(self) -> int: ...


StdIndex = Union[int, slice, "ellipsis", tuple[Union[slice, int, "ellipsis"], ...]]
StrIndex = Union[
    int, slice, str, "ellipsis", tuple[Union[slice, int, str, "ellipsis"], ...]
]


class RebinProtocol(Protocol):
    def axis_mapping(
        self, axis: PlottableAxis
    ) -> tuple[Sequence[int], PlottableAxis | None]: ...
