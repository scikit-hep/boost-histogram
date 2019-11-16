from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

__all__ = (
    "Storage",
    "Int",
    "Double",
    "AtomicInt",
    "Unlimited",
    "Weight",
    "Mean",
    "WeightedMean",
)


from ._internal.storage import (
    Storage,
    Int,
    Double,
    AtomicInt,
    Unlimited,
    Weight,
    Mean,
    WeightedMean,
)


# for lazy folks
int = Int()
double = Double()
unlimited = Unlimited()
atomic_int = AtomicInt()
weight = Weight()
mean = Mean()
weighted_mean = WeightedMean()
