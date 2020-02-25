from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from .._core import storage as store
from .utils import set_family, MAIN_FAMILY, CPP_FAMILY, set_module

# Simple mixin to provide a common base class for types
class Storage(object):
    def __repr__(self):
        return "{self.__class__.__name__}()".format(self=self)


### MAIN FAMILY


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


### CPP FAMILY ###


@set_family(CPP_FAMILY)
@set_module("boost_histogram.cpp.storage")
class int64(Int64):
    pass


@set_family(CPP_FAMILY)
@set_module("boost_histogram.cpp.storage")
class double(Double):
    pass


@set_family(CPP_FAMILY)
@set_module("boost_histogram.cpp.storage")
class atomic_int64(AtomicInt64):
    pass


@set_family(CPP_FAMILY)
@set_module("boost_histogram.cpp.storage")
class unlimited(Unlimited):
    pass


@set_family(CPP_FAMILY)
@set_module("boost_histogram.cpp.storage")
class weight(Weight):
    pass


@set_family(CPP_FAMILY)
@set_module("boost_histogram.cpp.storage")
class mean(Mean):
    pass


@set_family(CPP_FAMILY)
@set_module("boost_histogram.cpp.storage")
class weighted_mean(WeightedMean):
    pass
