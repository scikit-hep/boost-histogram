from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any, Callable, ClassVar, Literal

import numpy as np

from .accumulators import Mean, WeightedMean, WeightedSum
from .typing import ArrayLike, StrIndex, Ufunc

UFMethod = Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "at"]


class View(np.ndarray):
    __slots__ = ()
    _FIELDS: ClassVar[tuple[str, ...]]
    _PARENT: type[WeightedSum | WeightedMean | Mean]

    def __getitem__(self, ind: StrIndex) -> np.typing.NDArray[Any]:  # type: ignore[override]
        sliced = super().__getitem__(ind)

        # If the shape is empty, return the parent type
        if not sliced.shape:
            return self._PARENT._make(*sliced)

        # If the dtype has changed, return a normal array (no longer a record)
        if sliced.dtype != self.dtype:
            return np.asarray(sliced)

        # Otherwise, no change, return the same View type
        return sliced

    def __repr__(self) -> str:
        # NumPy starts the ndarray class name with "array", so we replace it
        # with our class name
        return f"{self.__class__.__name__}(\n      " + repr(self.view(np.ndarray))[6:]

    def __str__(self) -> str:
        my_fields = ", ".join(self._FIELDS)
        return f"{self.__class__.__name__}: ({my_fields})\n{self.view(np.ndarray)}"

    def __setitem__(self, ind: StrIndex, value: ArrayLike) -> None:  # type: ignore[override]
        # `.value` really is ["value"] for an record array
        if isinstance(ind, str):
            super().__setitem__(ind, value)
            return

        current_ndim = super().__getitem__(ind).ndim

        array: np.typing.NDArray[Any] = np.asarray(value)
        msg = "Needs matching ndarray or n+1 dim array"
        if array.ndim == current_ndim + 1:
            if len(self._FIELDS) == array.shape[-1]:
                self.__setitem__(ind, self._PARENT._array(*np.moveaxis(array, -1, 0)))
                return
            msg += f", final dimension should be {len(self._FIELDS)} for this storage, got {array.shape[-1]} instead"
            raise ValueError(msg)
        if self.dtype == array.dtype:
            super().__setitem__(ind, array)
            return

        msg += f", {current_ndim}D {self.dtype} or {current_ndim + 1}D required, got {array.ndim}D {array.dtype}"
        raise ValueError(msg)


def make_getitem_property(name: str) -> property:
    def fget(self: Mapping[str, Any]) -> Any:
        return self[name]

    def fset(self: MutableMapping[str, Any], value: Any) -> None:
        self[name] = value

    return property(fget, fset)


def fields(*names: str) -> Callable[[type[object]], type[object]]:
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

    def injector(cls: type[object]) -> type[object]:
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

    value: np.typing.NDArray[Any]
    variance: np.typing.NDArray[Any]

    # Could be implemented on master View
    def __array_ufunc__(
        self, ufunc: Ufunc, method: UFMethod, *inputs: Any, **kwargs: Any
    ) -> np.typing.NDArray[Any]:
        # Avoid infinite recursion
        raw_inputs = [np.asarray(x) for x in inputs]

        # Support unary + and -
        if (
            method == "__call__"
            and len(raw_inputs) == 1
            and ufunc in {np.negative, np.positive}
        ):
            (result,) = (
                kwargs.pop("out")
                if "out" in kwargs
                else [np.empty(self.shape, self.dtype)]
            )

            ufunc(raw_inputs[0]["value"], out=result["value"], **kwargs)
            result["variance"] = raw_inputs[0]["variance"]
            return result.view(self.__class__)

        if method == "__call__" and len(raw_inputs) == 2:
            (result,) = (
                kwargs.pop("out")
                if "out" in kwargs
                else [np.empty(self.shape, self.dtype)]
            )

            # Addition of two views
            if raw_inputs[0].dtype == raw_inputs[1].dtype:
                if ufunc in {np.add, np.subtract}:
                    ufunc(
                        raw_inputs[0]["value"],
                        raw_inputs[1]["value"],
                        out=result["value"],
                        **kwargs,
                    )
                    np.add(
                        raw_inputs[0]["variance"],
                        raw_inputs[1]["variance"],
                        out=result["variance"],
                        **kwargs,
                    )
                    return result.view(self.__class__)

                # If unsupported, just pass through (will return not implemented)
                # pylint: disable-next=no-member
                return super().__array_ufunc__(ufunc, method, *raw_inputs, **kwargs)

            # View with normal value or array
            if ufunc in {np.add, np.subtract}:
                if self.dtype == raw_inputs[0].dtype:
                    ufunc(
                        raw_inputs[0]["value"],
                        raw_inputs[1],
                        out=result["value"],
                        **kwargs,
                    )
                    np.add(
                        raw_inputs[0]["variance"],
                        raw_inputs[1] ** 2,
                        out=result["variance"],
                        **kwargs,
                    )
                else:
                    ufunc(
                        raw_inputs[0],
                        raw_inputs[1]["value"],
                        out=result["value"],
                        **kwargs,
                    )
                    np.add(
                        raw_inputs[0] ** 2,
                        raw_inputs[1]["variance"],
                        out=result["variance"],
                        **kwargs,
                    )
                return result.view(self.__class__)

            if ufunc in {np.multiply, np.divide, np.true_divide, np.floor_divide}:
                if self.dtype == raw_inputs[0].dtype:
                    ufunc(
                        raw_inputs[0]["value"],
                        raw_inputs[1],
                        out=result["value"],
                        **kwargs,
                    )
                    ufunc(
                        raw_inputs[0]["variance"],
                        raw_inputs[1] ** 2,
                        out=result["variance"],
                        **kwargs,
                    )
                else:
                    ufunc(
                        raw_inputs[0],
                        raw_inputs[1]["value"],
                        out=result["value"],
                        **kwargs,
                    )
                    ufunc(
                        raw_inputs[0] ** 2,
                        raw_inputs[1]["variance"],
                        out=result["variance"],
                        **kwargs,
                    )

                return result.view(self.__class__)

        # ufuncs that are allowed to reduce
        if ufunc in {np.add} and method == "reduce" and len(raw_inputs) == 1:
            results = (ufunc.reduce(self[field], **kwargs) for field in self._FIELDS)
            return self._PARENT._make(*results)

        # ufuncs that are allowed to accumulate
        if ufunc in {np.add} and method == "accumulate" and len(raw_inputs) == 1:
            (result,) = (
                kwargs.pop("out")
                if "out" in kwargs
                else [np.empty(self.shape, self.dtype)]
            )
            for field in self._FIELDS:
                ufunc.accumulate(self[field], out=result[field], **kwargs)
            return result.view(self.__class__)

        # If unsupported, just pass through (will return NotImplemented or things like == will work but not return subclasses)
        # pylint: disable-next=no-member
        return super().__array_ufunc__(ufunc, method, *raw_inputs, **kwargs)


@fields(
    "sum_of_weights",
    "sum_of_weights_squared",
    "value",
    "_sum_of_weighted_deltas_squared",
)
class WeightedMeanView(View):
    __slots__ = ()
    _PARENT = WeightedMean

    sum_of_weights: np.typing.NDArray[Any]
    sum_of_weights_squared: np.typing.NDArray[Any]
    value: np.typing.NDArray[Any]
    _sum_of_weighted_deltas_squared: np.typing.NDArray[Any]

    @property
    def variance(self) -> np.typing.NDArray[Any]:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self["_sum_of_weighted_deltas_squared"] / (
                self["sum_of_weights"]
                - self["sum_of_weights_squared"] / self["sum_of_weights"]
            )


@fields("count", "value", "_sum_of_deltas_squared")
class MeanView(View):
    __slots__ = ()
    _PARENT = Mean

    count: np.typing.NDArray[Any]
    value: np.typing.NDArray[Any]
    _sum_of_deltas_squared: np.typing.NDArray[Any]

    # Variance is a computation
    @property
    def variance(self) -> np.typing.NDArray[Any]:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self["_sum_of_deltas_squared"] / (self["count"] - 1)


def _to_view(
    item: np.typing.NDArray[Any], value: bool = False
) -> np.typing.NDArray[Any] | WeightedSumView | WeightedMeanView | MeanView:
    for cls in View.__subclasses__():
        if item.dtype.names == cls._FIELDS:
            ret = item.view(cls)
            if value and ret.shape:
                return ret.value
            return ret
    return item
