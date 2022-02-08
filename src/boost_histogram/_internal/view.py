from typing import Any, Callable, ClassVar, Mapping, MutableMapping, Tuple, Type, Union

import numpy as np

from ..accumulators import Mean, WeightedMean, WeightedSum
from .typing import ArrayLike, StrIndex, Ufunc


class View(np.ndarray):  # type: ignore[type-arg]
    __slots__ = ()
    _FIELDS: ClassVar[Tuple[str, ...]]

    def __getitem__(self, ind: StrIndex) -> "np.typing.NDArray[Any]":
        sliced = super().__getitem__(ind)

        # If the shape is empty, return the parent type
        if not sliced.shape:
            return self._PARENT._make(*sliced)  # type: ignore[attr-defined, no-any-return]

        # If the dtype has changed, return a normal array (no longer a record)
        if sliced.dtype != self.dtype:
            return np.asarray(sliced)

        # Otherwise, no change, return the same View type
        return sliced  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        # NumPy starts the ndarray class name with "array", so we replace it
        # with our class name
        return f"{self.__class__.__name__}(\n      " + repr(self.view(np.ndarray))[6:]

    def __str__(self) -> str:
        my_fields = ", ".join(self._FIELDS)
        return f"{self.__class__.__name__}: ({my_fields})\n{self.view(np.ndarray)}"

    def __setitem__(self, ind: StrIndex, value: ArrayLike) -> None:
        # `.value` really is ["value"] for an record array
        if isinstance(ind, str):
            super().__setitem__(ind, value)  # type: ignore[no-untyped-call]
            return

        array = np.asarray(value)
        if (
            array.ndim == super().__getitem__(ind).ndim + 1
            and len(self._FIELDS) == array.shape[-1]
        ):
            self.__setitem__(ind, self._PARENT._array(*np.moveaxis(array, -1, 0)))  # type: ignore[attr-defined]
        elif self.dtype == array.dtype:
            super().__setitem__(ind, array)  # type: ignore[no-untyped-call]
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
        my_fields = []
        for name in names:
            my_fields.append(name)
            setattr(cls, name, make_getitem_property(name))
            cls._FIELDS = tuple(my_fields)  # type: ignore[attr-defined]

        return cls

    return injector


@fields("value", "variance")
class WeightedSumView(View):
    __slots__ = ()
    _PARENT = WeightedSum

    value: "np.typing.NDArray[Any]"
    variance: "np.typing.NDArray[Any]"

    # Could be implemented on master View
    def __array_ufunc__(
        self, ufunc: Ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> "np.typing.NDArray[Any]":

        # This one is defined for record arrays, so just use it
        # (Doesn't get picked up the pass-through)
        if ufunc is np.equal and method == "__call__" and len(inputs) == 2:
            return ufunc(np.asarray(inputs[0]), np.asarray(inputs[1]), **kwargs)  # type: ignore[no-any-return]

        # Support unary + and -
        if (
            method == "__call__"
            and len(inputs) == 1
            and ufunc in {np.negative, np.positive}
        ):
            (result,) = kwargs.pop("out", [np.empty(self.shape, self.dtype)])

            ufunc(inputs[0]["value"], out=result["value"], **kwargs)
            result["variance"] = inputs[0]["variance"]
            return result.view(self.__class__)  # type: ignore[no-any-return]

        if method == "__call__" and len(inputs) == 2:
            input_0 = np.asarray(inputs[0])
            input_1 = np.asarray(inputs[1])

            (result,) = kwargs.pop("out", [np.empty(self.shape, self.dtype)])

            # Addition of two views
            if input_0.dtype == input_1.dtype:
                if ufunc in {np.add, np.subtract}:
                    ufunc(
                        input_0["value"],
                        input_1["value"],
                        out=result["value"],
                        **kwargs,
                    )
                    np.add(
                        input_0["variance"],
                        input_1["variance"],
                        out=result["variance"],
                        **kwargs,
                    )
                    return result.view(self.__class__)  # type: ignore[no-any-return]

                # If unsupported, just pass through (will return not implemented)
                return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)  # type: ignore[misc, no-any-return]

            # View with normal value or array
            if ufunc in {np.add, np.subtract}:
                if self.dtype == input_0.dtype:
                    ufunc(input_0["value"], input_1, out=result["value"], **kwargs)
                    np.add(
                        input_0["variance"],
                        input_1**2,
                        out=result["variance"],
                        **kwargs,
                    )
                else:
                    ufunc(input_0, input_1["value"], out=result["value"], **kwargs)
                    np.add(
                        input_0**2,
                        input_1["variance"],
                        out=result["variance"],
                        **kwargs,
                    )
                return result.view(self.__class__)  # type: ignore[no-any-return]

            if ufunc in {np.multiply, np.divide, np.true_divide, np.floor_divide}:
                if self.dtype == input_0.dtype:
                    ufunc(input_0["value"], input_1, out=result["value"], **kwargs)
                    ufunc(
                        input_0["variance"],
                        input_1**2,
                        out=result["variance"],
                        **kwargs,
                    )
                else:
                    ufunc(input_0, input_1["value"], out=result["value"], **kwargs)
                    ufunc(
                        input_0**2,
                        input_1["variance"],
                        out=result["variance"],
                        **kwargs,
                    )

                return result.view(self.__class__)  # type: ignore[no-any-return]

        # ufuncs that are allowed to reduce
        if ufunc in {np.add} and method == "reduce" and len(inputs) == 1:
            results = (ufunc.reduce(self[field], **kwargs) for field in self._FIELDS)
            return self._PARENT._make(*results)  # type: ignore[no-any-return]

        # ufuncs that are allowed to accumulate
        if ufunc in {np.add} and method == "accumulate" and len(inputs) == 1:
            (result,) = kwargs.pop("out", [np.empty(self.shape, self.dtype)])
            for field in self._FIELDS:
                ufunc.accumulate(self[field], out=result[field], **kwargs)
            return result.view(self.__class__)  # type: ignore[no-any-return]

        # If unsupported, just pass through (will return not implemented)
        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)  # type: ignore[misc, no-any-return]


@fields(
    "sum_of_weights",
    "sum_of_weights_squared",
    "value",
    "_sum_of_weighted_deltas_squared",
)
class WeightedMeanView(View):
    __slots__ = ()
    _PARENT = WeightedMean

    sum_of_weights: "np.typing.NDArray[Any]"
    sum_of_weights_squared: "np.typing.NDArray[Any]"
    value: "np.typing.NDArray[Any]"
    _sum_of_weighted_deltas_squared: "np.typing.NDArray[Any]"

    @property
    def variance(self) -> "np.typing.NDArray[Any]":
        with np.errstate(divide="ignore", invalid="ignore"):
            return self["_sum_of_weighted_deltas_squared"] / (  # type: ignore[no-any-return]
                self["sum_of_weights"]
                - self["sum_of_weights_squared"] / self["sum_of_weights"]
            )


@fields("count", "value", "_sum_of_deltas_squared")
class MeanView(View):
    __slots__ = ()
    _PARENT = Mean

    count: "np.typing.NDArray[Any]"
    value: "np.typing.NDArray[Any]"
    _sum_of_deltas_squared: "np.typing.NDArray[Any]"

    # Variance is a computation
    @property
    def variance(self) -> "np.typing.NDArray[Any]":
        with np.errstate(divide="ignore", invalid="ignore"):
            return self["_sum_of_deltas_squared"] / (self["count"] - 1)


def _to_view(
    item: "np.typing.NDArray[Any]", value: bool = False
) -> Union["np.typing.NDArray[Any]", WeightedSumView, WeightedMeanView, MeanView]:
    for cls in View.__subclasses__():
        if cls._FIELDS == item.dtype.names:
            ret = item.view(cls)
            if value and ret.shape:
                return ret.value  # type: ignore[no-any-return,attr-defined]
            return ret
    return item
