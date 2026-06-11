from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np

from .accumulators import Mean, WeightedMean, WeightedSum
from .typing import ArrayLike, StrIndex, Ufunc

UFMethod = Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "at"]


class View(np.ndarray[Any, Any]):
    __slots__ = ()
    _FIELDS: ClassVar[tuple[str, ...]]
    _PARENT: type[WeightedSum | WeightedMean | Mean]

    # Whether ``a - b`` (and unary ``-``) and multiplying/dividing two views are
    # meaningful for this view. Overridden per subclass.
    _SUPPORTS_SUB: ClassVar[bool] = False
    _SUPPORTS_VIEW_PRODUCT: ClassVar[bool] = False

    def __getitem__(self, ind: StrIndex) -> np.typing.NDArray[Any]:  # type: ignore[override]
        sliced = super().__getitem__(ind)  # type: ignore[index]

        # If the dtype has changed, return a normal array (no longer a record).
        # This must be checked before the shape check so that 0-d field access
        # (e.g. ``view["value"]`` on a 0-d MeanView) returns a plain scalar
        # array instead of trying to iterate over a 0-d array.
        if sliced.dtype != self.dtype:
            return np.asarray(sliced)

        # If the shape is empty, return the parent type
        if not sliced.shape:
            return self._PARENT._make(*sliced)  # type: ignore[return-value]

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

        current_ndim = super().__getitem__(ind).ndim  # type: ignore[index]

        array: np.typing.NDArray[Any] = np.asarray(value)
        msg = "Needs matching ndarray or n+1 dim array"
        if array.ndim == current_ndim + 1:
            if len(self._FIELDS) == array.shape[-1]:
                self.__setitem__(ind, self._PARENT._array(*np.moveaxis(array, -1, 0)))  # type: ignore[assignment]
                return
            msg += f", final dimension should be {len(self._FIELDS)} for this storage, got {array.shape[-1]} instead"
            raise ValueError(msg)
        if self.dtype == array.dtype:
            super().__setitem__(ind, array)  # type: ignore[index]
            return

        msg += f", {current_ndim}D {self.dtype} or {current_ndim + 1}D required, got {array.ndim}D {array.dtype}"
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # Arithmetic: shared ufunc scaffolding that dispatches the field math
    # to per-view hooks (``_combine``/``_add_scalar``/``_scale``/...).
    # ------------------------------------------------------------------
    def __array_ufunc__(
        self, ufunc: Ufunc, method: UFMethod, *inputs: Any, **kwargs: Any
    ) -> np.typing.NDArray[Any]:
        # Avoid infinite recursion by working with plain ndarrays
        raw_inputs = [np.asarray(x) for x in inputs]

        # Reductions (``.sum()`` / ``np.add.reduce``) don't get a pre-allocated
        # ``out`` here; the hook builds the reduced accumulator(s) itself. A
        # caller-supplied ``out`` cannot be forwarded to the per-field math (it
        # would be reused for every field), so it is filled in afterwards.
        match method, ufunc, raw_inputs:
            case "reduce", np.add, [raw_input]:
                out_tuple = kwargs.pop("out", None)
                reduced = self._reduce(raw_input, **kwargs)
                if out_tuple is None:
                    return reduced
                out_array: np.typing.NDArray[Any] = out_tuple[0]
                if out_array.dtype != self.dtype:
                    raise TypeError(
                        f"out= with dtype {out_array.dtype} is not supported for "
                        f"{self.__class__.__name__} reductions; the dtype must "
                        f"match the view ({self.dtype})"
                    )
                out_array[...] = reduced
                return out_array.view(self.__class__)

        # Everything else fills a pre-allocated result of this view's dtype
        (result,) = (
            kwargs.pop("out")
            if "out" in kwargs
            else [
                np.empty(
                    np.broadcast_shapes(*(np.shape(r) for r in raw_inputs)),
                    self.dtype,
                )
            ]
        )
        match method, ufunc, raw_inputs:
            # Unary + and -
            case "__call__", (np.negative | np.positive), [raw_input]:
                self._unary(raw_input, result, ufunc, **kwargs)
                return result.view(self.__class__)

            # Addition and subtraction
            case "__call__", (np.add | np.subtract), [raw1, raw2]:
                if ufunc is np.subtract and not self._SUPPORTS_SUB:
                    raise TypeError(
                        f"{self.__class__.__name__} does not support subtraction"
                    )
                if raw1.dtype == raw2.dtype:
                    self._combine(raw1, raw2, result, ufunc, **kwargs)
                else:
                    self._add_scalar(raw1, raw2, result, ufunc, **kwargs)
                return result.view(self.__class__)

            # Multiplication and division
            case (
                "__call__",
                (np.multiply | np.divide | np.floor_divide),
                [raw1, raw2],
            ):
                if raw1.dtype == raw2.dtype:
                    if not self._SUPPORTS_VIEW_PRODUCT:
                        raise TypeError(
                            f"{self.__class__.__name__} does not support multiplying "
                            "or dividing two views"
                        )
                    self._product(raw1, raw2, result, ufunc, **kwargs)
                else:
                    self._scale(raw1, raw2, result, ufunc, **kwargs)
                return result.view(self.__class__)

            # Cumulative sums (``np.cumsum`` / ``np.add.accumulate``)
            case "accumulate", np.add, [raw_input]:
                self._accumulate(raw_input, result, **kwargs)
                return result.view(self.__class__)

        # If unsupported, just pass through (will return not implemented)
        # pylint: disable-next=no-member
        return super().__array_ufunc__(ufunc, method, *raw_inputs, **kwargs)  # type: ignore[no-any-return]

    # --- Arithmetic hooks; the defaults are the field-wise (linear) behavior
    # correct for additive storages like WeightedSum. Non-linear views (means)
    # override the ones they need and reject the rest. ---

    def _unary(self, raw: Any, out: Any, ufunc: Ufunc, **kwargs: Any) -> None:
        del raw, out, kwargs
        raise TypeError(
            f"{self.__class__.__name__} does not support unary {ufunc.__name__}"
        )

    def _combine(
        self, raw1: Any, raw2: Any, out: Any, ufunc: Ufunc, **kwargs: Any
    ) -> None:
        del raw1, raw2, out, ufunc, kwargs
        raise TypeError(f"{self.__class__.__name__} does not support this operation")

    def _add_scalar(
        self, raw1: Any, raw2: Any, out: Any, ufunc: Ufunc, **kwargs: Any
    ) -> None:
        del raw1, raw2, out, kwargs
        raise TypeError(
            f"{self.__class__.__name__} does not support {ufunc.__name__} "
            "with a scalar or array"
        )

    def _scale(
        self, raw1: Any, raw2: Any, out: Any, ufunc: Ufunc, **kwargs: Any
    ) -> None:
        del raw1, raw2, out, kwargs
        raise TypeError(f"{self.__class__.__name__} does not support {ufunc.__name__}")

    def _product(
        self, raw1: Any, raw2: Any, out: Any, ufunc: Ufunc, **kwargs: Any
    ) -> None:
        del raw1, raw2, out, ufunc, kwargs
        raise TypeError(
            f"{self.__class__.__name__} does not support multiplying two views"
        )

    def _reduce(self, raw: Any, **kwargs: Any) -> np.typing.NDArray[Any]:
        results = (np.add.reduce(raw[field], **kwargs) for field in self._FIELDS)
        return self._PARENT._make(*results)  # type: ignore[return-value]

    def _accumulate(self, raw: Any, out: Any, **kwargs: Any) -> None:
        for field in self._FIELDS:
            np.add.accumulate(raw[field], out=out[field], **kwargs)


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
    _SUPPORTS_SUB = True

    value: np.typing.NDArray[Any]
    variance: np.typing.NDArray[Any]

    def _unary(self, raw: Any, out: Any, ufunc: Ufunc, **kwargs: Any) -> None:
        ufunc(raw["value"], out=out["value"], **kwargs)
        out["variance"] = raw["variance"]

    def _combine(
        self, raw1: Any, raw2: Any, out: Any, ufunc: Ufunc, **kwargs: Any
    ) -> None:
        # Values add/subtract; independent variances always add.
        ufunc(raw1["value"], raw2["value"], out=out["value"], **kwargs)
        np.add(raw1["variance"], raw2["variance"], out=out["variance"], **kwargs)

    def _add_scalar(
        self, raw1: Any, raw2: Any, out: Any, ufunc: Ufunc, **kwargs: Any
    ) -> None:
        # The scalar/array operand is treated as a constant with variance s**2.
        if raw1.dtype == self.dtype:
            ufunc(raw1["value"], raw2, out=out["value"], **kwargs)
            np.add(raw1["variance"], raw2**2, out=out["variance"], **kwargs)
        else:
            ufunc(raw1, raw2["value"], out=out["value"], **kwargs)
            np.add(raw1**2, raw2["variance"], out=out["variance"], **kwargs)

    def _scale(
        self, raw1: Any, raw2: Any, out: Any, ufunc: Ufunc, **kwargs: Any
    ) -> None:
        # Multiplying/dividing by a scalar scales the variance by s**2.
        if raw1.dtype == self.dtype:
            ufunc(raw1["value"], raw2, out=out["value"], **kwargs)
            ufunc(raw1["variance"], raw2**2, out=out["variance"], **kwargs)
        elif ufunc in {np.divide, np.floor_divide}:
            # ``scalar / view``: the reciprocal of a weighted sum is not a
            # linear scaling, so no meaningful variance can be propagated.
            raise TypeError(
                f"Cannot divide a scalar or array by a {self.__class__.__name__}"
            )
        else:
            ufunc(raw1, raw2["value"], out=out["value"], **kwargs)
            ufunc(raw1**2, raw2["variance"], out=out["variance"], **kwargs)


class _MeanArithmetic:
    """Combine + scalar-scaling arithmetic shared by the mean-like views.

    Addition combines two samples (the Welford/West parallel merge, mirroring the
    C++ ``mean``/``weighted_mean`` ``operator+=``); multiplication/division by a
    scalar scales the mean and ``_sum_of_*deltas_squared`` (by ``s`` and ``s**2``
    respectively) while leaving the weights untouched (mirroring ``operator*=``).
    Subtraction, scalar addition, unary minus, and cumulative sums are undefined
    for means and fall through to the rejecting :class:`View` defaults.
    """

    __slots__ = ()

    # Field names supplied by the concrete view subclass.
    _WEIGHT_FIELD: ClassVar[str]
    _DELTAS_FIELD: ClassVar[str]
    _WEIGHT_SQ_FIELD: ClassVar[str | None] = None

    if TYPE_CHECKING:
        # Provided by the View base that the concrete subclass also inherits.
        _FIELDS: ClassVar[tuple[str, ...]]
        _PARENT: type[Mean | WeightedMean]
        dtype: np.dtype[Any]

    def _combine(
        self, raw1: Any, raw2: Any, out: Any, ufunc: Ufunc, **kwargs: Any
    ) -> None:
        # Means only ever reach here for addition (subtraction is rejected).
        del ufunc, kwargs
        weight, deltas, weight_sq = (
            self._WEIGHT_FIELD,
            self._DELTAS_FIELD,
            self._WEIGHT_SQ_FIELD,
        )
        n1, n2 = raw1[weight], raw2[weight]
        mu1, mu2 = raw1["value"], raw2["value"]

        total = n1 + n2
        # New mean = weight-weighted average; 0 (not NaN) where both are empty.
        new_value = np.zeros_like(total)
        np.divide(n1 * mu1 + n2 * mu2, total, out=new_value, where=total != 0)

        # ``deltas`` uses the *new* value; empty operands contribute nothing.
        # Compute every field into a fresh array *before* writing, so an
        # in-place ``v += other`` (where ``out`` aliases ``raw1``) is correct.
        new_deltas = (
            raw1[deltas]
            + raw2[deltas]
            + n1 * (new_value - mu1) ** 2
            + n2 * (new_value - mu2) ** 2
        )
        new_weight_sq = None if weight_sq is None else raw1[weight_sq] + raw2[weight_sq]

        out[weight] = total
        out["value"] = new_value
        out[deltas] = new_deltas
        if weight_sq is not None:
            out[weight_sq] = new_weight_sq

    def _scale(
        self, raw1: Any, raw2: Any, out: Any, ufunc: Ufunc, **kwargs: Any
    ) -> None:
        del kwargs
        # Floor division would truncate the (floating-point) mean and deltas, so
        # it is not a meaningful scaling of a mean accumulator.
        if ufunc is np.floor_divide:
            raise TypeError(
                f"{self.__class__.__name__} does not support floor division"
            )
        weight, deltas, weight_sq = (
            self._WEIGHT_FIELD,
            self._DELTAS_FIELD,
            self._WEIGHT_SQ_FIELD,
        )
        if raw1.dtype == self.dtype:
            view, scalar = raw1, raw2
        elif ufunc is np.divide:
            # ``scalar / view`` is not a meaningful scaling of a mean.
            raise TypeError(
                f"Cannot divide a scalar or array by a {self.__class__.__name__}"
            )
        else:
            view, scalar = raw2, raw1

        out[weight] = view[weight]
        out["value"] = ufunc(view["value"], scalar)
        # The mean scales by ``s``; its variance (hence ``deltas``) scales by
        # ``s**2``. Applying ``ufunc`` twice gives exactly that: for multiply
        # ``(x*s)*s == x*s**2`` and for divide ``(x/s)/s == x/s**2``.
        out[deltas] = ufunc(ufunc(view[deltas], scalar), scalar)
        if weight_sq is not None:
            out[weight_sq] = view[weight_sq]

    def _reduce(self, raw: Any, **kwargs: Any) -> np.typing.NDArray[Any]:
        axis = kwargs.pop("axis", 0)
        keepdims = kwargs.pop("keepdims", False)
        # ``np.sum`` always forwards ``dtype=None`` and ``where=True``; any
        # other reduction argument (a real dtype, a where mask, initial, ...)
        # cannot be honored by the mean-merging math below, so reject it
        # instead of silently ignoring it.
        if (
            kwargs.pop("dtype", None) is not None
            or kwargs.pop("where", True) is not True
            or kwargs
        ):
            raise TypeError(
                f"{self.__class__.__name__} reductions only support the "
                "axis and keepdims arguments"
            )
        weight, deltas, weight_sq = (
            self._WEIGHT_FIELD,
            self._DELTAS_FIELD,
            self._WEIGHT_SQ_FIELD,
        )
        counts = raw[weight]
        values = raw["value"]

        def reduce(array: Any, *, keep: bool = keepdims) -> Any:
            return np.add.reduce(array, axis=axis, keepdims=keep)  # type: ignore[call-overload]

        with np.errstate(divide="ignore", invalid="ignore"):
            total = reduce(counts)
            combined = np.where(total != 0, reduce(counts * values) / total, 0.0)

            # The spread term needs the combined mean broadcast back against the
            # un-reduced ``values``; ``keepdims=True`` keeps the reduced axis as
            # size 1 so this works for any ``axis`` (int, tuple, or None) and is
            # independent of the caller's ``keepdims``.
            total_b = reduce(counts, keep=True)
            mean_b = np.where(
                total_b != 0, reduce(counts * values, keep=True) / total_b, 0.0
            )
            new_deltas = reduce(raw[deltas]) + reduce(counts * (values - mean_b) ** 2)

        out_fields: dict[str, Any] = {
            weight: total,
            "value": combined,
            deltas: new_deltas,
        }
        if weight_sq is not None:
            out_fields[weight_sq] = reduce(raw[weight_sq])
        return self._PARENT._make(*(out_fields[name] for name in self._FIELDS))  # type: ignore[return-value]

    def _accumulate(self, raw: Any, out: Any, **kwargs: Any) -> None:
        del raw, out, kwargs
        raise TypeError(f"{self.__class__.__name__} does not support cumulative sums")


@fields(
    "sum_of_weights",
    "sum_of_weights_squared",
    "value",
    "_sum_of_weighted_deltas_squared",
)
class WeightedMeanView(_MeanArithmetic, View):
    __slots__ = ()
    _PARENT = WeightedMean

    _WEIGHT_FIELD = "sum_of_weights"
    _DELTAS_FIELD = "_sum_of_weighted_deltas_squared"
    _WEIGHT_SQ_FIELD = "sum_of_weights_squared"

    sum_of_weights: np.typing.NDArray[Any]
    sum_of_weights_squared: np.typing.NDArray[Any]
    value: np.typing.NDArray[Any]
    _sum_of_weighted_deltas_squared: np.typing.NDArray[Any]

    @property
    def variance(self) -> np.typing.NDArray[Any]:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self["_sum_of_weighted_deltas_squared"] / (  # type: ignore[no-any-return]
                self["sum_of_weights"]
                - self["sum_of_weights_squared"] / self["sum_of_weights"]
            )


@fields("count", "value", "_sum_of_deltas_squared")
class MeanView(_MeanArithmetic, View):
    __slots__ = ()
    _PARENT = Mean

    _WEIGHT_FIELD = "count"
    _DELTAS_FIELD = "_sum_of_deltas_squared"
    _WEIGHT_SQ_FIELD = None

    count: np.typing.NDArray[Any]
    value: np.typing.NDArray[Any]
    _sum_of_deltas_squared: np.typing.NDArray[Any]

    # Variance is a computation
    @property
    def variance(self) -> np.typing.NDArray[Any]:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self["_sum_of_deltas_squared"] / (self["count"] - 1)  # type: ignore[no-any-return]


def _to_view(
    item: np.typing.NDArray[Any], value: bool = False
) -> np.typing.NDArray[Any] | WeightedSumView | WeightedMeanView | MeanView:
    for cls in View.__subclasses__():
        if item.dtype.names == cls._FIELDS:
            ret = item.view(cls)
            if value and ret.shape:
                return ret.value  # type: ignore[no-any-return, attr-defined]
            return ret
    return item
