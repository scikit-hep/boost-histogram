from __future__ import annotations

from ._core.accumulators import (  # pylint: disable=import-error,no-name-in-module
    Mean,
    Sum,
    WeightedMean,
    WeightedSum,
)
from .typing import Accumulator

__all__ = ("Accumulator", "Mean", "Sum", "WeightedMean", "WeightedSum")

for cls in (Sum, Mean, WeightedSum, WeightedMean):
    cls.__module__ = "boost_histogram.accumulators"
del cls
