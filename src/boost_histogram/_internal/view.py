from typing import Any, Callable, ClassVar, Mapping, MutableMapping, Tuple, Type, Union

import numpy as np

from ..accumulators import Mean, WeightedMean, WeightedSum
from .typing import ArrayLike, StrIndex, Ufunc


class View(np.ndarray):
    __slots__ = ()
    _FIELDS: ClassVar[Tuple[str, ...]]

    def __getitem__(self, ind: StrIndex) -> np.ndarray:
        sliced = super().__getitem__(ind)

        # If the shape is empty, return the parent type
        if not sliced.shape:
            return self._PARENT._make(*sliced)  # type: ignore
        # If the dtype has changed, return a normal array (no longer a record)
        elif sliced.dtype != self.dtype:
            return np.asarray(sliced)
        # Otherwise, no change, return the same View type
        else:
            return sliced  # type: ignore

    def __repr__(self) -> str:
        # Numpy starts the ndarray class name with "array", so we replace it
        # with our class name
        return f"{self.__class__.__name__}(\n      " + repr(self.view(np.ndarray))[6:]

    def __str__(self) -> str:
        fields = ", ".join(self._FIELDS)
        return "{self.__class__.__name__}: ({fields})\n{arr}".format(
            self=self, fields=fields, arr=self.view(np.ndarray)
        )

    def __setitem__(self, ind: StrIndex, value: ArrayLike) -> None:
        # `.value` really is ["value"] for an record array
        if isinstance(ind, str):
            super().__setitem__(ind, value)  # type: ignore
            return

        array = np.asarray(value)
        if (
            array.ndim == super().__getitem__(ind).ndim + 1
            and len(self._FIELDS) == array.shape[-1]
        ):
            self.__setitem__(ind, self._PARENT._array(*np.moveaxis(array, -1, 0)))  # type: ignore
        elif self.dtype == array.dtype:
            super().__setitem__(ind, array)  # type: ignore
        else:
            raise ValueError("Needs matching ndarray or n+1 dim array")


def make_getitem_property(name: str) -> property:
    def fget(self: Mapping[str, Any]) -> Any:
        return self[name]

    def fset(self: MutableMapping[str, Any], value: Any) -> None:
        self[name] = value

    return property(fget, fset)


def fields(*names: str) -> Callable[[Type[object]], Type[object]]:
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

    def injector(cls: Type[object]) -> Type[object]:
        if hasattr(cls, "_FIELDS"):
            raise RuntimeError(
                f"{cls.__name__} already has had a fields decorator applied"
            )
        fields = []
        for name in names:
            fields.append(name)
            setattr(cls, name, make_getitem_property(name))
            cls._FIELDS = tuple(fields)  # type: ignore

        return cls

    return injector


@fields("value", "variance")
class WeightedSumView(View):
    __slots__ = ()
    _PARENT = WeightedSum

    value: np.ndarray
    variance: np.ndarray

    # Could be implemented on master View
    def __array_ufunc__(
        self, ufunc: Ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> np.ndarray:

        # This one is defined for record arrays, so just use it
        # (Doesn't get picked up the pass-through)
        if ufunc is np.equal and method == "__call__" and len(inputs) == 2:
            return ufunc(np.asarray(inputs[0]), np.asarray(inputs[1]), **kwargs)  # type: ignore

        # Support unary + and -
        if (
            method == "__call__"
            and len(inputs) == 1
            and ufunc in {np.negative, np.positive}
        ):
            (result,) = kwargs.pop("out", [np.empty(self.shape, self.dtype)])

            ufunc(inputs[0]["value"], out=result["value"], **kwargs)
            result["variance"] = inputs[0]["variance"]
            return result.view(self.__class__)  # type: ignore

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
                        **kwargs,
                    )
                    ufunc(
                        input_0["variance"],
                        input_1["variance"],
                        out=result["variance"],
                        **kwargs,
                    )
                    return result.view(self.__class__)  # type: ignore

            # View with normal value or array
            else:
                if ufunc in {np.add, np.subtract}:
                    if self.dtype == input_0.dtype:
                        ufunc(input_0["value"], input_1, out=result["value"], **kwargs)
                        np.add(
                            input_0["variance"],
                            input_1 ** 2,
                            out=result["variance"],
                            **kwargs,
                        )
                    else:
                        ufunc(input_0, input_1["value"], out=result["value"], **kwargs)
                        np.add(
                            input_0 ** 2,
                            input_1["variance"],
                            out=result["variance"],
                            **kwargs,
                        )
                    return result.view(self.__class__)  # type: ignore

                elif ufunc in {np.multiply, np.divide, np.true_divide, np.floor_divide}:
                    if self.dtype == input_0.dtype:
                        ufunc(input_0["value"], input_1, out=result["value"], **kwargs)
                        ufunc(
                            input_0["variance"],
                            input_1 ** 2,
                            out=result["variance"],
                            **kwargs,
                        )
                    else:
                        ufunc(input_0, input_1["value"], out=result["value"], **kwargs)
                        ufunc(
                            input_0 ** 2,
                            input_1["variance"],
                            out=result["variance"],
                            **kwargs,
                        )

                    return result.view(self.__class__)  # type: ignore

        # ufuncs that are allowed to reduce
        if ufunc in {np.add} and method == "reduce" and len(inputs) == 1:
            results = (ufunc.reduce(self[field], **kwargs) for field in self._FIELDS)
            return self._PARENT._make(*results)  # type: ignore

        # If unsupported, just pass through (will return not implemented)
        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)  # type: ignore


@fields(
    "sum_of_weights",
    "sum_of_weights_squared",
    "value",
    "_sum_of_weighted_deltas_squared",
)
class WeightedMeanView(View):
    __slots__ = ()
    _PARENT = WeightedMean

    sum_of_weights: np.ndarray
    sum_of_weights_squared: np.ndarray
    value: np.ndarray
    _sum_of_weighted_deltas_squared: np.ndarray

    @property
    def variance(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self["_sum_of_weighted_deltas_squared"] / (  # type: ignore
                self["sum_of_weights"]
                - self["sum_of_weights_squared"] / self["sum_of_weights"]
            )


@fields("count", "value", "_sum_of_deltas_squared")
class MeanView(View):
    __slots__ = ()
    _PARENT = Mean

    count: np.ndarray
    value: np.ndarray
    _sum_of_deltas_squared: np.ndarray

    # Variance is a computation
    @property
    def variance(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self["_sum_of_deltas_squared"] / (self["count"] - 1)  # type: ignore


def _to_view(
    item: np.ndarray, value: bool = False
) -> Union[np.ndarray, WeightedSumView, WeightedMeanView, MeanView]:
    for cls in View.__subclasses__():
        if cls._FIELDS == item.dtype.names:
            ret = item.view(cls)
            if value and ret.shape:
                return ret.value  # type: ignore
            else:
                return ret  # type: ignore
    return item
