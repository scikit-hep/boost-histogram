from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

__all__ = ("Sum", "Mean", "WeightedSum", "WeightedMean")

from ._core.accumulators import Sum, Mean, WeightedSum, WeightedMean

for cls in (Sum, Mean, WeightedSum, WeightedMean):
    cls.__module__ = "boost_histogram.accumulators"
del cls
