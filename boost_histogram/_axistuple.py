from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from .axis import Axis

import numpy as np


class AxesTuple(tuple):
    __slots__ = ()
    _MGRIDOPTS = {"sparse": True, "indexing": "ij"}

    @property
    def size(self):
        return tuple(s.size for s in self)

    @property
    def metadata(self):
        return tuple(s.metadata for s in self)

    @property
    def extent(self):
        return tuple(s.extent for s in self)

    @property
    def centers(self):
        gen = (s.centers for s in self)
        return np.meshgrid(*gen, **self._MGRIDOPTS)

    @property
    def edges(self):
        gen = (s.edges for s in self)
        return np.meshgrid(*gen, **self._MGRIDOPTS)

    @property
    def widths(self):
        gen = (s.widths for s in self)
        return np.meshgrid(*gen, **self._MGRIDOPTS)

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

    value.__doc__ = Axis.value.__doc__
    index.__doc__ = Axis.index.__doc__
    bin.__doc__ = Axis.bin.__doc__
