from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

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


class DepStorageMixin(object):
    @classmethod
    def _get_storage_(cls):
        warnings.warn("Use Int instead", DeprecationWarning)
        return cls._STORAGE()


class int(DepStorageMixin, Int):
    pass


class double(DepStorageMixin, Double):
    pass


class unlimited(DepStorageMixin, Unlimited):
    pass


class atomic_int(DepStorageMixin, AtomicInt):
    pass


class weight(DepStorageMixin, Weight):
    pass


class mean(DepStorageMixin, Mean):
    pass


class weighted_mean(DepStorageMixin, WeightedMean):
    pass
