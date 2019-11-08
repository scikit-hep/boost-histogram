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

for cls in (Storage, Int, Double, AtomicInt, Unlimited, Weight, Mean, WeightedMean):
    cls.__module__ = "boost_histogram.storage"


# Option 1:

# for lazy folks
int = Int()
double = Double()
unlimited = Unlimited()
atomic_int = AtomicInt()
weight = Weight()
mean = Mean()
weighted_mean = WeightedMean()

# Option 2

# class DepStorageMixin(object):
#     def _get_storage_(self):
#         import warnings
#         warnings.warn("Use Int instead", DeprecationWarning)
#         return cls._STORAGE()
#
#
# class int(DepStorageMixin, Int):
#     pass
#
#
# class double(DepStorageMixin, Double):
#     pass
#
#
# class unlimited(DepStorageMixin, Unlimited):
#     pass
#
#
# class atomic_int(DepStorageMixin, AtomicInt):
#     pass
#
#
# class weight(DepStorageMixin, Weight):
#     pass
#
#
# class mean(DepStorageMixin, Mean):
#     pass
#
#
# class weighted_mean(DepStorageMixin, WeightedMean):
#     pass
