from __future__ import annotations

from ._core.accumulators import (  # pylint: disable=import-error,no-name-in-module
    Mean,
    Sum,
    Values,
    WeightedMean,
    WeightedSum,
    WeightedValues,
)
from .typing import Accumulator

__all__ = (
    "Accumulator",
    "Mean",
    "Sum",
    "Values",
    "WeightedMean",
    "WeightedSum",
    "WeightedValues",
)

for cls in (Sum, Mean, WeightedSum, WeightedMean, Values, WeightedValues):
    cls.__module__ = "boost_histogram.accumulators"
del cls

# Not supported by pybind builtins
# Enable if wrapper added
# inject_signature("self, value")(Sum.fill)
# inject_signature("self, value, *, variance=None")(WeightedSum.fill)
# inject_signature("self, value, *, weight=None")(Mean.fill)
# inject_signature("self, value, *, weight=None")(WeightedMean.fill)
