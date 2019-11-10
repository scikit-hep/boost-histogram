from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from .._core import storage as store
from .utils import register

# Simple mixin to provide a common base class for types
# and nice reprs
class Storage(object):
    def __repr__(self):
        return "{self.__class__.__name__}()".format(self=self)


@register(store.int)
class Int(store.int, Storage):
    pass


@register(store.double)
class Double(store.double, Storage):
    pass


@register(store.atomic_int)
class AtomicInt(store.atomic_int, Storage):
    pass


@register(store.unlimited)
class Unlimited(store.unlimited, Storage):
    pass


@register(store.weight)
class Weight(store.weight, Storage):
    pass


@register(store.mean)
class Mean(store.mean, Storage):
    pass


@register(store.weighted_mean)
class WeightedMean(store.weighted_mean, Storage):
    pass
