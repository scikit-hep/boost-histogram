from ._core.accumulators import Mean, Sum, WeightedMean, WeightedSum
from ._internal.typing import Accumulator

__all__ = ("Sum", "Mean", "WeightedSum", "WeightedMean", "Accumulator")

for cls in (Sum, Mean, WeightedSum, WeightedMean):
    cls.__module__ = "boost_histogram.accumulators"
del cls

# Not supported by pybind builtins
# Enable if wrapper added
# inject_signature("self, value")(Sum.fill)
# inject_signature("self, value, *, variance=None")(WeightedSum.fill)
# inject_signature("self, value, *, weight=None")(Mean.fill)
# inject_signature("self, value, *, weight=None")(WeightedMean.fill)
