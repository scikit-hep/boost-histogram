from __future__ import absolute_import, division, print_function

from ..accumulators import Mean, WeightedMean, WeightedSum

import numpy as np


class View(np.ndarray):
    __slots__ = ()

    def __getitem__(self, ind):
        sliced = super(View, self).__getitem__(ind)

        # If the shape is empty, return the parent type
        if not sliced.shape:
            return self._PARENT._make(*sliced)
        # If the dtype has changed, return a normal array (no longer a record)
        elif sliced.dtype != self.dtype:
            return np.asarray(sliced)
        # Otherwise, no change, return the same View type
        else:
            return sliced

    def __repr__(self):
        # Numpy starts the ndarray class name with "array", so we replace it
        # with our class name
        return (
            "{self.__class__.__name__}(\n      ".format(self=self)
            + repr(self.view(np.ndarray))[6:]
        )

    def __str__(self):
        fields = ", ".join(self._FIELDS)
        return "{self.__class__.__name__}: ({fields})\n{arr}".format(
            self=self, fields=fields, arr=self.view(np.ndarray)
        )


def fields(*names):
    """
    This decorator adds the name to the _FIELDS
    class property (for printing in reprs), and
    adds a property that looks like this:

    @property
    def name(self):
        return self["name"]
    @name.setter
    def name(self, value):
        self["name"] = value
    """

    def injector(cls):
        if hasattr(cls, "_FIELDS"):
            raise RuntimeError(
                "{0} already has had a fields decorator applied".format(
                    self.__class__.__name__
                )
            )
        fields = []
        for name in names:
            fields.append(name)

            def fget(self):
                return self[name]

            def fset(self, value):
                self[name] = value

            setattr(cls, name, property(fget, fset))
            cls._FIELDS = tuple(fields)
        return cls

    return injector


@fields("value", "variance")
class WeightedSumView(View):
    __slots__ = ()
    _PARENT = WeightedSum


@fields(
    "sum_of_weights",
    "sum_of_weights_squared",
    "value",
    "sum_of_weighted_deltas_squared",
)
class WeightedMeanView(View):
    __slots__ = ()
    _PARENT = WeightedMean

    @property
    def variance(self):
        return self["sum_of_weighted_deltas_squared"] / (
            self["sum_of_weights"]
            - self["sum_of_weights_squared"] / self["sum_of_weights"]
        )


@fields("count", "value", "sum_of_deltas_squared")
class MeanView(View):
    __slots__ = ()
    _PARENT = Mean

    # Variance is a computation
    @property
    def variance(self):
        return self["sum_of_deltas_squared"] / (self["count"] - 1)


def _to_view(item, value=False):
    for cls in View.__subclasses__():
        if cls._FIELDS == item.dtype.names:
            ret = item.view(cls)
            if value and ret.shape:
                return ret.value
            else:
                return ret
    return item
