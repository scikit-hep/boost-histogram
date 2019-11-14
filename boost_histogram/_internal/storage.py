from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from .._core import storage as store
from .utils import register, set_family, MAIN_FAMILY, CPP_FAMILY, set_module

# Simple mixin to provide a common base class for types
class Storage(object):
    def __repr__(self):
        return "{self.__class__.__name__}()".format(self=self)


### MAIN FAMILY


@register({store.int})
@set_family(MAIN_FAMILY)
@set_module("boost_histogram.storage")
class Int(store.int, Storage):
    pass


@register({store.double})
@set_family(MAIN_FAMILY)
@set_module("boost_histogram.storage")
class Double(store.double, Storage):
    pass


@register({store.atomic_int})
@set_family(MAIN_FAMILY)
@set_module("boost_histogram.storage")
class AtomicInt(store.atomic_int, Storage):
    pass


@register({store.unlimited})
@set_family(MAIN_FAMILY)
@set_module("boost_histogram.storage")
class Unlimited(store.unlimited, Storage):
    pass


@register({store.weight})
@set_family(MAIN_FAMILY)
@set_module("boost_histogram.storage")
class Weight(store.weight, Storage):
    pass


@register({store.mean})
@set_family(MAIN_FAMILY)
@set_module("boost_histogram.storage")
class Mean(store.mean, Storage):
    pass


@register({store.weighted_mean})
@set_family(MAIN_FAMILY)
@set_module("boost_histogram.storage")
class WeightedMean(store.weighted_mean, Storage):
    pass


### CPP FAMILY ###


@set_family(CPP_FAMILY)
@set_module("boost_histogram.cpp.storage")
class int(Int):
    pass


@set_family(CPP_FAMILY)
@set_module("boost_histogram.cpp.storage")
class double(Double):
    pass


@set_family(CPP_FAMILY)
@set_module("boost_histogram.cpp.storage")
class atomic_int(AtomicInt):
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
