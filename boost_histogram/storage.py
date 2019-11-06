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


def int():
    warnings.warn("Use Int instead", DeprecationWarning)
    return Int()


def double():
    warnings.warn("Use Double instead", DeprecationWarning)
    return Double()


def unlimited():
    warnings.warn("Use Unlimited instead", DeprecationWarning)
    return Unlimited()


def atomic_int():
    warnings.warn("Use AtomicInt instead", DeprecationWarning)
    return AtomicInt()


def weight():
    warnings.warn("Use Weight instead", DeprecationWarning)
    return Weight()


def mean():
    warnings.warn("Use Mean instead", DeprecationWarning)
    return Mean()


def weighted_mean():
    warnings.warn("Use WeightedMean instead", DeprecationWarning)
    return WeightedMean()
