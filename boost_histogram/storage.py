from __future__ import absolute_import

del absolute_import

import warnings

from ._internal.storage import (
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
