from typing import ClassVar, Type, Union

import boost_histogram

from .._core import accumulators
from .._core import storage as store
from .utils import set_module


# Simple mixin to provide a common base class for types
class Storage:
    _family: object

    def __init_subclass__(cls, *, family: object) -> None:
        super().__init_subclass__()
        cls._family = family

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    accumulator: ClassVar[
        Union[
            Type[int],
            Type[float],
            Type[accumulators.WeightedMean],
            Type[accumulators.WeightedSum],
            Type[accumulators.Mean],
        ]
    ]


@set_module("boost_histogram.storage")
class Int64(store.int64, Storage, family=boost_histogram):
    accumulator = int


@set_module("boost_histogram.storage")
class Double(store.double, Storage, family=boost_histogram):
    accumulator = float


@set_module("boost_histogram.storage")
class AtomicInt64(store.atomic_int64, Storage, family=boost_histogram):
    accumulator = int


@set_module("boost_histogram.storage")
class Unlimited(store.unlimited, Storage, family=boost_histogram):
    accumulator = float


@set_module("boost_histogram.storage")
class Weight(store.weight, Storage, family=boost_histogram):
    accumulator = accumulators.WeightedSum


@set_module("boost_histogram.storage")
class Mean(store.mean, Storage, family=boost_histogram):
    accumulator = accumulators.Mean


@set_module("boost_histogram.storage")
class WeightedMean(store.weighted_mean, Storage, family=boost_histogram):
    accumulator = accumulators.WeightedMean
