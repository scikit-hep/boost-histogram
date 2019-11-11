from __future__ import absolute_import, division, print_function

from ..accumulators import Mean, WeightedMean, WeightedSum

import numpy as np


class View(np.ndarray):
    __slots__ = ()

    def __getitem__(self, ind):
        sliced = super(View, self).__getitem__(ind)
        if sliced.shape:
            return sliced
        else:
            return self._PARENT._make(*sliced)

    def __repr__(self):
        return repr(self.view(np.ndarray))

    def __str__(self):
        return str(self.view(np.ndarray))


class WeightedSumView(View):
    __slots__ = ()
    _PARENT = WeightedSum
    _FIELDS = ("value", "variance")

    @property
    def value(self):
        return self["value"]

    @property
    def variance(self):
        return self["variance"]


class WeightedMeanView(View):
    __slots__ = ()
    _PARENT = WeightedMean
    _FIELDS = (
        "sum_of_weights",
        "sum_of_weights_squared",
        "value",
        "sum_of_weighted_deltas_squared",
    )

    @property
    def sum_of_weights(self):
        return self["sum_of_weights"]

    @property
    def sum_of_weights_squared(self):
        return self["sum_of_weights_squared"]

    @property
    def value(self):
        return self["value"]

    @property
    def sum_of_weighted_deltas_squared(self):
        return self["sum_of_weighted_deltas_squared"]

    @property
    def variance(self):
        return self["sum_of_weighted_deltas_squared"] / (
            self["sum_of_weights"]
            - self["sum_of_weights_squared"] / self["sum_of_weights"]
        )


def _to_view(item, value=False):
    for cls in View.__subclasses__():
        if cls._FIELDS == item.dtype.names:
            ret = item.view(cls)
            if value and ret.shape:
                return ret.value
            else:
                return ret
    return item


class MeanView(View):
    __slots__ = ()
    _PARENT = Mean
    _FIELDS = ("count", "value", "sum_of_deltas_squared")

    @property
    def count(self):
        return self["count"]

    @property
    def value(self):
        return self["value"]

    @property
    def sum_of_deltas_squared(self):
        return self["sum_of_deltas_squared"]

    # Variance is a computation
    @property
    def variance(self):
        return self["sum_of_deltas_squared"] / (self["count"] - 1)
