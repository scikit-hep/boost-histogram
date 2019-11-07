from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from .._core import storage as store


class Storage(object):
    __slots__ = "_storage"

    def __init__(self):
        self._storage = self._STORAGE()

    def __eq__(self, other):
        return self._storage == other._storage

    def _get_storage_(self):
        return self._storage

    def __repr__(self):
        return "{self.__class__.__name__}()".format(self=self)


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
    """Get a storage class from C++ class"""
    for base in Storage.__subclasses__():
        if st == base._STORAGE:
            return base

    raise TypeError("Invalid storage passed in")
