from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from .._core import storage as store
from .utils import _walk_subclasses


class Storage(object):
    __slots__ = ()

    def __eq__(self, other):
        return self.__class__ == other.__class__

    def __ne__(self, other):
        return self.__class__ != other.__class__

    # Override this to allow configurable storages
    @classmethod
    def _get_storage_(cls):
        return cls._STORAGE()


class Int(Storage):
    _STORAGE = store.int


class Double(Storage):
    _STORAGE = store.double


class AtomicInt(Storage):
    _STORAGE = store.atomic_int


class Unlimited(Storage):
    _STORAGE = store.unlimited


class Weight(Storage):
    _STORAGE = store.weight


class Mean(Storage):
    _STORAGE = store.mean


class WeightedMean(Storage):
    _STORAGE = store.weighted_mean


def _to_storage(st):
    for base in _walk_subclasses(Storage):
        if st == base._STORAGE:
            return base

    raise TypeError("Invalid storage passed in")
