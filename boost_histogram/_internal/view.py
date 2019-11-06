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


class MeanView(View):
    __slots__ = ()
    _PARENT = Mean
    _FIELDS = ("sum_", "mean_", "sum_of_deltas_squared_")

    @property
    def count(self):
        return self["mean_"]

    @property
    def value(self):
        return self["sum_"]

    # Variance is a computation
    @property
    def variance(self):
        return self["sum_of_deltas_squared_"] / (self["sum_"] - 1)


class WeightedSumView(View):
    __slots__ = ()
    _PARENT = WeightedSum
    _FIELDS = ("sum_of_weights_", "sum_of_weights_squared_")

    @property
    def sum_of_weights(self):
        return self["sum_of_weights_"]

    @property
    def sum_of_weights_squared(self):
        return self["sum_of_weights_squared_"]


class WeightedMeanView(View):
    __slots__ = ()
    _PARENT = WeightedMean
    _FIELDS = (
        "sum_of_weights_",
        "sum_of_weights_squared_",
        "weighted_mean_",
        "sum_of_weighted_deltas_squared_",
    )

    @property
    def sum_of_weights(self):
        return self["sum_of_weights_"]

    @property
    def sum_of_weights_squared(self):
        return self["sum_of_weights_squared_"]

    @property
    def value(self):
        return self["weighted_mean_"]

    @property
    def variance(self):
        return self["sum_of_weighted_deltas_squared_"] / (
            self["sum_of_weights_"]
            - self["sum_of_weights_squared_"] / self["sum_of_weights_"]
        )


def _to_view(item):
    for cls in View.__subclasses__():
        if cls._FIELDS == item.dtype.fields:
            return item.view(cls)
    return item
