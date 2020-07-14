# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from .axis import Axis
from .utils import set_module

import numpy as np

del absolute_import, division, print_function


@set_module("boost_histogram.axis")
class ArrayTuple(tuple):
    __slots__ = ()

    def __getattr__(self, name):
        return self.__class__(getattr(a, name) for a in self)

    def __dir__(self):
        return sorted(dir(self.__class__) + dir(np.ndarray))

    def __call__(self, *args, **kwargs):
        return self.__class__(a(*args, **kwargs) for a in self)

    def broadcast(self):
        """
        The arrays in this tuple will be compressed if possible to save memory.
        Use this method to broadcast them out into their full memory
        representation.
        """
        return self.__class__(np.broadcast_arrays(*self))


@set_module("boost_histogram.axis")
class AxesTuple(tuple):
    __slots__ = ()
    _MGRIDOPTS = {"sparse": True, "indexing": "ij"}

    @property
    def size(self):
        return tuple(s.size for s in self)

    @property
    def metadata(self):
        return tuple(s.metadata for s in self)

    @metadata.setter
    def metadata(self, values):
        for s, v in zip(self, values):
            s.metadata = v

    @property
    def extent(self):
        return tuple(s.extent for s in self)

    @property
    def centers(self):
        gen = (s.centers for s in self)
        return ArrayTuple(np.meshgrid(*gen, **self._MGRIDOPTS))

    @property
    def edges(self):
        gen = (s.edges for s in self)
        return ArrayTuple(np.meshgrid(*gen, **self._MGRIDOPTS))

    @property
    def widths(self):
        gen = (s.widths for s in self)
        return ArrayTuple(np.meshgrid(*gen, **self._MGRIDOPTS))

    def value(self, *indexes):
        if len(indexes) != len(self):
            raise IndexError(
                "Must have the same number of arguments as the number of axes"
            )
        return tuple(self[i].value(indexes[i]) for i in range(len(indexes)))

    def bin(self, *indexes):
        if len(indexes) != len(self):
            raise IndexError(
                "Must have the same number of arguments as the number of axes"
            )
        return tuple(self[i].bin(indexes[i]) for i in range(len(indexes)))

    def index(self, *values):
        if len(values) != len(self):
            raise IndexError(
                "Must have the same number of arguments as the number of axes"
            )
        return tuple(self[i].index(values[i]) for i in range(len(values)))

    def __getitem__(self, item):
        result = super(AxesTuple, self).__getitem__(item)
        return self.__class__(result) if isinstance(result, tuple) else result

    # Python 2 support - remove after 1.0
    def __getslice__(self, start, stop):
        result = super(AxesTuple, self).__getslice__(start, stop)
        return self.__class__(result)

    value.__doc__ = Axis.value.__doc__
    index.__doc__ = Axis.index.__doc__
    bin.__doc__ = Axis.bin.__doc__
