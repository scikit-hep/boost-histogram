from __future__ import annotations

from typing import ClassVar

import boost_histogram

from ._core import accumulators  # pylint: disable=no-name-in-module
from ._core import storage as store  # pylint: disable=no-name-in-module

__all__ = [
    "AtomicInt64",
    "Double",
    "DoubleSparse",
    "Int64",
    "Mean",
    "MultiCell",
    "Storage",
    "Unlimited",
    "Weight",
    "WeightedMean",
]


def __dir__() -> list[str]:
    return __all__


# Simple mixin to provide a common base class for types
class Storage:
    _family: object

    def __init_subclass__(cls, *, family: object) -> None:
        super().__init_subclass__()
        cls._family = family

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    accumulator: ClassVar[
        (
            type[
                int
                | float
                | accumulators.WeightedMean
                | accumulators.WeightedSum
                | accumulators.Mean
            ]
        )
    ]


class Int64(store.int64, Storage, family=boost_histogram):
    accumulator = int


class Double(store.double, Storage, family=boost_histogram):
    accumulator = float


class DoubleSparse(store.double_sparse, Storage, family=boost_histogram):
    """
    Sparse storage of doubles: only filled cells are kept (in a hash map), so
    histograms over very large or high-dimensional axis spaces stay small in
    memory. It does not support ``.view()`` / dense-array access; use
    :meth:`Histogram.to_coo` to read the filled cells instead.
    """

    accumulator = float


class AtomicInt64(store.atomic_int64, Storage, family=boost_histogram):
    accumulator = int


class Unlimited(store.unlimited, Storage, family=boost_histogram):
    accumulator = float


class Weight(store.weight, Storage, family=boost_histogram):
    accumulator = accumulators.WeightedSum


class Mean(store.mean, Storage, family=boost_histogram):
    accumulator = accumulators.Mean


class WeightedMean(store.weighted_mean, Storage, family=boost_histogram):
    accumulator = accumulators.WeightedMean


class MultiCell(store.multi_cell, Storage, family=boost_histogram):
    accumulator = float

    __match_args__ = ("nelem",)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.nelem})"
