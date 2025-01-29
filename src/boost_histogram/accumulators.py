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

# Not supported by pybind builtins
# Enable if wrapper added
# inject_signature("self, value")(Sum.fill)
# inject_signature("self, value, *, variance=None")(WeightedSum.fill)
# inject_signature("self, value, *, weight=None")(Mean.fill)
# inject_signature("self, value, *, weight=None")(WeightedMean.fill)
