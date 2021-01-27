# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from .._core import storage as store
from .utils import MAIN_FAMILY, set_family, set_module

del absolute_import, division, print_function


# Simple mixin to provide a common base class for types
class Storage(object):
    def __repr__(self):
        return "{self.__class__.__name__}()".format(self=self)


# MAIN FAMILY


@set_family(MAIN_FAMILY)
@set_module("boost_histogram.storage")
class Int64(store.int64, Storage):
    pass


@set_family(MAIN_FAMILY)
@set_module("boost_histogram.storage")
class Double(store.double, Storage):
    pass


@set_family(MAIN_FAMILY)
@set_module("boost_histogram.storage")
class AtomicInt64(store.atomic_int64, Storage):
    pass


@set_family(MAIN_FAMILY)
@set_module("boost_histogram.storage")
class Unlimited(store.unlimited, Storage):
    pass


@set_family(MAIN_FAMILY)
@set_module("boost_histogram.storage")
class Weight(store.weight, Storage):
    pass


@set_family(MAIN_FAMILY)
@set_module("boost_histogram.storage")
class Mean(store.mean, Storage):
    pass


@set_family(MAIN_FAMILY)
@set_module("boost_histogram.storage")
class WeightedMean(store.weighted_mean, Storage):
    pass
