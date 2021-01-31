# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np

from ..accumulators import Mean, WeightedMean, WeightedSum


class View(np.ndarray):
    __slots__ = ()

    def __getitem__(self, ind):
        sliced = super(View, self).__getitem__(ind)

        # If the shape is empty, return the parent type
        if not sliced.shape:
            return self._PARENT._make(*sliced)  # type: ignore
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
        fields = ", ".join(self._FIELDS)  # type: ignore
        return "{self.__class__.__name__}: ({fields})\n{arr}".format(
            self=self, fields=fields, arr=self.view(np.ndarray)
        )

    def __setitem__(self, ind, value):
        # `.value` really is ["value"] for an record array
        if isinstance(ind, str):
            super(View, self).__setitem__(ind, value)
            return

        array = np.asarray(value)
        if (
            array.ndim == super(View, self).__getitem__(ind).ndim + 1
            and len(self._FIELDS) == array.shape[-1]  # type: ignore
        ):
            self.__setitem__(ind, self._PARENT._array(*np.moveaxis(array, -1, 0)))  # type: ignore
        elif self.dtype == array.dtype:
            super(View, self).__setitem__(ind, array)
        else:
            raise ValueError("Needs matching ndarray or n+1 dim array")


def make_getitem_property(name):
    def fget(self):
        return self[name]

    def fset(self, value):
        self[name] = value

    return property(fget, fset)


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
                "{} already has had a fields decorator applied".format(cls.__name__)
            )
        fields = []
        for name in names:
            fields.append(name)
            setattr(cls, name, make_getitem_property(name))
            cls._FIELDS = tuple(fields)

        return cls

    return injector


@fields("value", "variance")
class WeightedSumView(View):
    __slots__ = ()
    _PARENT = WeightedSum

    # Could be implemented on master View
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        # This one is defined for record arrays, so just use it
        # (Doesn't get picked up the pass-through)
        if ufunc is np.equal and method == "__call__" and len(inputs) == 2:
            return ufunc(np.asarray(inputs[0]), np.asarray(inputs[1]), **kwargs)

        # Support unary + and -
        if method == "__call__" and len(inputs) == 1:
            if ufunc in {np.negative, np.positive}:
                (result,) = kwargs.pop("out", [np.empty(self.shape, self.dtype)])

                ufunc(inputs[0]["value"], out=result["value"], **kwargs)
                result["variance"] = inputs[0]["variance"]
                return result.view(self.__class__)

        if method == "__call__" and len(inputs) == 2:
            input_0 = np.asarray(inputs[0])
            input_1 = np.asarray(inputs[1])

            (result,) = kwargs.pop("out", [np.empty(self.shape, self.dtype)])

            # Addition of two views
            if input_0.dtype == input_1.dtype:
                if ufunc in {np.add}:
                    ufunc(
                        input_0["value"],
                        input_1["value"],
                        out=result["value"],
                        **kwargs
                    )
                    ufunc(
                        input_0["variance"],
                        input_1["variance"],
                        out=result["variance"],
                        **kwargs
                    )
                    return result.view(self.__class__)

            # View with normal value or array
            else:
                if ufunc in {np.add, np.subtract}:
                    if self.dtype == input_0.dtype:
                        ufunc(input_0["value"], input_1, out=result["value"], **kwargs)
                        np.add(
                            input_0["variance"],
                            input_1 ** 2,
                            out=result["variance"],
                            **kwargs
                        )
                    else:
                        ufunc(input_0, input_1["value"], out=result["value"], **kwargs)
                        np.add(
                            input_0 ** 2,
                            input_1["variance"],
                            out=result["variance"],
                            **kwargs
                        )
                    return result.view(self.__class__)

                elif ufunc in {np.multiply, np.divide, np.true_divide, np.floor_divide}:
                    if self.dtype == input_0.dtype:
                        ufunc(input_0["value"], input_1, out=result["value"], **kwargs)
                        ufunc(
                            input_0["variance"],
                            np.abs(input_1),
                            out=result["variance"],
                            **kwargs
                        )
                    else:
                        ufunc(input_0, input_1["value"], out=result["value"], **kwargs)
                        ufunc(
                            np.abs(input_0),
                            input_1["variance"],
                            out=result["variance"],
                            **kwargs
                        )

                    return result.view(self.__class__)

        # ufuncs that are allowed to reduce
        if ufunc in {np.add} and method == "reduce" and len(inputs) == 1:
            results = (ufunc.reduce(self[field], **kwargs) for field in self._FIELDS)  # type: ignore
            return self._PARENT._make(*results)

        # If unsupported, just pass through (will return not implemented)
        return super(WeightedSumView, self).__array_ufunc__(  # type: ignore
            ufunc, method, *inputs, **kwargs
        )


@fields(
    "sum_of_weights",
    "sum_of_weights_squared",
    "value",
    "_sum_of_weighted_deltas_squared",
)
class WeightedMeanView(View):
    __slots__ = ()
    _PARENT = WeightedMean

    @property
    def variance(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            return self["_sum_of_weighted_deltas_squared"] / (
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
        with np.errstate(divide="ignore", invalid="ignore"):
            return self["sum_of_deltas_squared"] / (self["count"] - 1)


def _to_view(item, value=False):
    for cls in View.__subclasses__():
        if cls._FIELDS == item.dtype.names:  # type: ignore
            ret = item.view(cls)
            if value and ret.shape:
                return ret.value
            else:
                return ret
    return item
