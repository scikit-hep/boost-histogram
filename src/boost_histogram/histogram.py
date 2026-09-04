from __future__ import annotations

import collections.abc
import copy
import enum
import logging
import threading
import typing
import warnings
from collections.abc import Callable, Iterable, Mapping
from os import cpu_count
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    NewType,
    SupportsIndex,
    TypeAlias,
    TypeVar,
)

import numpy as np

import boost_histogram
from boost_histogram import _core

from . import serialization
from . import storage as bhs
from ._compat.typing import Self
from ._utils import cast, register
from .axis import AxesTuple, Axis, Variable
from .storage import Double, Storage
from .view import MeanView, WeightedMeanView, WeightedSumView, _to_view

if TYPE_CHECKING:
    from types import EllipsisType

    from .typing import (
        Accumulator,
        ArrayLike,
        CppHistogram,
        Mean,
        RebinProtocol,
        WeightedMean,
        WeightedSum,
    )


# This is a StrEnum as defined in Python 3.11
class Kind(str, enum.Enum):
    COUNT = "COUNT"
    MEAN = "MEAN"

    __str__ = str.__str__


__all__ = [
    "Histogram",
    "IndexingExpr",
    "Kind",
]


def __dir__() -> list[str]:
    return __all__


NOTHING = object()


_histograms: set[type[CppHistogram]] = {
    _core.hist.any_double,
    _core.hist.any_int64,
    _core.hist.any_atomic_int64,
    _core.hist.any_unlimited,
    _core.hist.any_weight,
    _core.hist.any_mean,
    _core.hist.any_weighted_mean,
    _core.hist.any_multi_cell,
}

# Tuple form of ``_histograms`` for fast isinstance checks. The set above is
# fixed at import time (only ``@register`` reads it), so this never goes stale.
_histogram_types: tuple[type[CppHistogram], ...] = tuple(_histograms)

logger = logging.getLogger(__name__)

# User-facing operator symbols for the in-place dunders dispatched in
# _compute_inplace_op, used to build clear errors when a storage's underlying
# C++ histogram does not support an operation between two histograms.
_INPLACE_OP_SYMBOLS: dict[str, str] = {
    "__iadd__": "+",
    "__isub__": "-",
    "__imul__": "*",
    "__itruediv__": "/",
}


CppAxis = NewType("CppAxis", object)

SimpleIndexing: TypeAlias = (
    "SupportsIndex | slice | RebinProtocol | np.typing.NDArray[Any]"
)
InnerIndexing: TypeAlias = "SimpleIndexing | Callable[[Axis], int]"
FullInnerIndexing: TypeAlias = "InnerIndexing | list[InnerIndexing]"
IndexingWithMapping: TypeAlias = "FullInnerIndexing | Mapping[int, FullInnerIndexing]"
IndexingExpr: TypeAlias = (
    "IndexingWithMapping | tuple[IndexingWithMapping, ...] | EllipsisType"
)

T = TypeVar("T")


IntHists = TypeVar(
    "IntHists", bound="Histogram[bhs.AtomicInt64] | Histogram[bhs.Int64]"
)
FloatHists = TypeVar(
    "FloatHists", bound="Histogram[bhs.Double] | Histogram[bhs.Unlimited]"
)
ListHists = TypeVar("ListHists", bound="Histogram[bhs.MultiCell]")
WeightHists = TypeVar("WeightHists", bound="Histogram[bhs.Weight]")
MeanHists = TypeVar("MeanHists", bound="Histogram[bhs.Mean]")
WeightedMeanHists = TypeVar("WeightedMeanHists", bound="Histogram[bhs.WeightedMean]")


@typing.overload
def _fill_cast(
    value: tuple[T, ...] | list[T], *, inner: Literal[False] = False
) -> tuple[T | np.typing.NDArray[Any], ...]: ...


@typing.overload
def _fill_cast(value: T, *, inner: bool = False) -> T | np.typing.NDArray[Any]: ...


def _fill_cast(
    value: Any, *, inner: bool = False
) -> Any | np.typing.NDArray[Any] | tuple[Any | np.typing.NDArray[Any], ...]:
    """
    Convert to NumPy arrays. Some buffer objects do not get converted by forcecast.
    If not called by itself (inner=False), then will work through one level of tuple/list.
    """
    if value is None or isinstance(value, (str, bytes)):
        return value

    if not inner and isinstance(value, (tuple, list)):
        return tuple(_fill_cast(a, inner=True) for a in value)

    if hasattr(value, "__iter__") or hasattr(value, "__array__"):
        return np.asarray(value)

    return value


def mean_storage_sample_check(sample: ArrayLike | None) -> None:
    if sample is None:
        raise TypeError("Sample key-argument (sample=) needs to be provided.")
    msg1 = f"Sample key-argument needs to be a number or a sequence, {sample.__class__.__name__} given."
    if isinstance(sample, str):
        raise TypeError(msg1)
    sample_dim = np.ndim(sample)
    msg2 = f"Sample key-argument needs to be a scalar or 1 dimensional, {sample_dim} given."
    if sample_dim > 1:
        raise ValueError(msg2)


def _arg_shortcut(item: tuple[int, float, float] | Axis | CppAxis) -> CppAxis:
    if isinstance(item, tuple) and len(item) == 3:
        msg = "Using () directly in constructor is a developer shortcut and will be removed in a future version"
        warnings.warn(msg, FutureWarning, stacklevel=4)
        return _core.axis.regular_uoflow(item[0], item[1], item[2])  # type: ignore[return-value]

    if isinstance(item, Axis):
        return item._ax  # type: ignore[no-any-return]

    raise TypeError("Only axes supported in histogram constructor")


# Discrete C++ axes that ``axis::edges`` represents as 0..size (the category
# axes); everything else gets its true (continuous-like) edge values.
_category_cpp_axes = (
    _core.axis.category_int,
    _core.axis.category_int_growth,
    _core.axis.category_int_none,
    _core.axis.category_str,
    _core.axis.category_str_growth,
    _core.axis.category_str_none,
)

# Axes that the C++ ``axis::edges`` helper does not nudge when producing
# NumPy-convention (upper-edge inclusive) edges.
_no_nudge_cpp_axes = (
    _core.axis.regular_none,
    _core.axis.regular_uflow,
)


def _numpy_compatible_edges(cpp_ax: Any, flow: bool) -> np.typing.NDArray[np.float64]:
    """
    Edges for one C++ axis following the NumPy convention (upper edge
    inclusive); this replicates exactly what the C++ ``axis::edges(ax, flow,
    true)`` helper produces, without requiring a copy of the bin contents.
    """
    if isinstance(cpp_ax, _category_cpp_axes):
        overflow = int(flow and cpp_ax.traits_overflow)
        return np.arange(cpp_ax.size + 1 + overflow, dtype=np.float64)

    underflow = int(flow and cpp_ax.traits_underflow)
    overflow = int(flow and cpp_ax.traits_overflow)

    edges: np.typing.NDArray[np.float64] = cpp_ax.edges
    if underflow or overflow:
        full = np.empty(len(edges) + underflow + overflow, dtype=np.float64)
        full[underflow : underflow + len(edges)] = edges
        if underflow:
            full[0] = cpp_ax.value(-1)
        if overflow:
            full[-1] = cpp_ax.value(cpp_ax.size + 1)
        edges = full

    if not isinstance(cpp_ax, _no_nudge_cpp_axes):
        last = cpp_ax.size + underflow
        edges[last] = np.nextafter(edges[last], -np.inf)

    return edges


def _expand_ellipsis(indexes: Iterable[Any], rank: int) -> list[Any]:
    indexes = list(indexes)
    # Compare by identity: ``==`` is ambiguous when indexes contain NumPy arrays.
    ellipsis_positions = [i for i, ind in enumerate(indexes) if ind is Ellipsis]
    number_ellipses = len(ellipsis_positions)
    if number_ellipses == 0:
        return indexes
    if number_ellipses == 1:
        index = ellipsis_positions[0]
        additional = rank + 1 - len(indexes)
        if additional < 0:
            raise IndexError("too many indices for histogram")

        # Fill out the ellipsis with empty slices
        return indexes[:index] + [slice(None)] * additional + indexes[index + 1 :]

    raise IndexError("an index can only have a single ellipsis ('...')")


def _combine_group_contents(
    new_view: np.typing.NDArray[Any],
    reduced_view: np.typing.NDArray[Any],
    i: int,
    j: int,
    jj: int,
) -> None:
    """
    Add bin ``j`` (along view dimension ``i``) of ``reduced_view`` into bin
    ``jj`` of ``new_view``, in-place. Used for rebinning with groups.
    """
    pos = [slice(None)] * (i)
    if new_view.dtype.names:
        for field in new_view.dtype.names:
            new_view[(*pos, jj, ...)][field] += reduced_view[(*pos, j, ...)][field]  # type: ignore[arg-type]
    else:
        new_view[(*pos, jj, ...)] += reduced_view[(*pos, j, ...)]  # type: ignore[arg-type]


H = TypeVar("H", bound="Histogram[Any]")
S = TypeVar("S", bound="Storage")

NO_METADATA = object()


# We currently do not cast *to* a histogram, but this is consistent
# and could be used later.
@register(_histograms)  # type: ignore[arg-type]
class Histogram(typing.Generic[S]):
    # Note this is a __slots__ __dict__ class!
    __slots__ = (
        "__dict__",
        "_hist",
        "axes",
    )
    # .metadata and ._variance_known are part of the dict.
    # .metadata will not be placed in the dict if not passed.

    _family: ClassVar[object] = boost_histogram

    axes: AxesTuple
    _hist: CppHistogram
    _variance_known: bool

    def __init_subclass__(cls, *, family: object | None = None) -> None:
        """
        Sets the family for the histogram. This should be a unique object (such
        as the main module of your package) that is consistently set across all
        subclasses. When converting back from C++, casting will try to always
        pick the best matching family from the loaded subclasses for Axis and
        such.
        """
        super().__init_subclass__()
        cls._family = family if family is not None else object()

    @typing.overload
    def __init__(
        self, arg: Histogram[S], /, *, metadata: Any = ..., __dict__: Any = ...
    ) -> None: ...

    @typing.overload
    def __init__(
        self, arg: dict[str, Any], /, *, metadata: Any = ..., __dict__: Any = ...
    ) -> None: ...

    @typing.overload
    def __init__(
        self, arg: CppHistogram, /, *, metadata: Any = ..., __dict__: Any = ...
    ) -> None: ...

    @typing.overload
    def __init__(
        self,
        *axes: Axis | CppAxis,
        storage: S,
        metadata: Any = ...,
        __dict__: Any = ...,
    ) -> None: ...

    @typing.overload
    def __init__(
        self: Histogram[Double],
        *axes: Axis | CppAxis,
        storage: None = ...,
        metadata: Any = ...,
        __dict__: Any = ...,
    ) -> None: ...

    def __init__(
        self,
        *axes: Axis | CppAxis | Histogram[Any] | CppHistogram | dict[str, Any],
        storage: S | None = None,
        metadata: Any = NO_METADATA,
        __dict__: Any = None,
    ) -> None:
        """
        Construct a new histogram.

        If you pass in a single argument, this will be treated as a
        histogram and this will convert the histogram to this type of
        histogram.

        Parameters
        ----------
        *args : Axis
            Provide 1 or more axis instances.
        storage : Storage = bh.storage.Double()
            Select a storage to use in the histogram
        metadata : Any = None
            Data that is passed along if a new histogram is created. No not use
            in new code; use ``__dict__`` instead.
        __dict__ : Any = None
            Better way to set metadata.
        """
        self._variance_known = True
        storage_err_msg = "storage= is not allowed with conversion constructor"

        if metadata is not NO_METADATA and __dict__:
            msg = (
                "Can't set both metadata and __dict__. Set the 'metadata' key instead."
            )
            raise TypeError(msg)
        if metadata is not NO_METADATA:
            __dict__ = {"metadata": metadata}
        if __dict__ is None:
            __dict__ = {}

        # Allow construction from a raw histogram object (internal)
        if len(axes) == 1 and isinstance(axes[0], _histogram_types):
            if storage is not None:
                raise TypeError(storage_err_msg)
            cpp_hist: CppHistogram = axes[0]
            self._from_histogram_cpp(cpp_hist, __dict__=__dict__)
            return

        # If we construct with another Histogram as the only positional argument,
        # support that too
        if len(axes) == 1 and isinstance(axes[0], Histogram):
            if storage is not None:
                raise TypeError(storage_err_msg)
            self._from_histogram_object(axes[0], __dict__=__dict__)
            return

        # Support objects that provide a to_boost method, like Uproot
        if len(axes) == 1 and hasattr(axes[0], "_to_boost_histogram_"):
            if storage is not None:
                raise TypeError(storage_err_msg)
            self._from_histogram_object(
                axes[0]._to_boost_histogram_(), __dict__=__dict__
            )
            return

        # Support UHI
        if len(axes) == 1 and isinstance(axes[0], dict) and "uhi_schema" in axes[0]:
            if storage is not None:
                raise TypeError(storage_err_msg)
            self._from_histogram_object(
                serialization.from_uhi(axes[0]), __dict__=__dict__
            )
            return

        resolved_storage = Double() if storage is None else storage

        self.__dict__.update(__dict__)

        # Check for missed parenthesis or incorrect types
        if not isinstance(resolved_storage, Storage):
            if isinstance(resolved_storage, type) and issubclass(  # type: ignore[unreachable]
                resolved_storage, Storage
            ):
                msg = f"Storages need to be initialized; use {resolved_storage.__name__}() instead. Please add ()."
                raise TypeError(msg)
            msg = f"Only storages allowed in storage argument, got {resolved_storage!r}"
            raise TypeError(msg)

        # Allow a tuple to represent a regular axis
        axes = tuple(_arg_shortcut(arg) for arg in axes)  # type: ignore[arg-type]

        if len(axes) > _core.hist._axes_limit:
            msg = f"Too many axes, must be less than {_core.hist._axes_limit}"
            raise IndexError(msg)

        # Check all available histograms, and if the storage matches, return that one
        for h in _histograms:
            if isinstance(resolved_storage, h._storage_type):
                self._hist = h(axes, resolved_storage)  # type: ignore[arg-type]
                self.axes = self._generate_axes_()
                return

        raise TypeError("Unsupported storage")

    @classmethod
    def _clone(
        cls,
        _hist: Histogram[Any] | CppHistogram,
        *,
        other: Histogram[Any] | None = None,
        memo: Any = NOTHING,
    ) -> Self:
        """
        Clone a histogram (possibly of a different base). Does not trigger __init__.
        This will copy data from `other=` if non-None, otherwise metadata gets copied from the input.
        """

        self = cls.__new__(cls)
        if isinstance(_hist, _histogram_types):
            self._from_histogram_cpp(_hist, __dict__={})
            if other is not None:
                return cls._clone(self, other=other, memo=memo)
            return self

        assert isinstance(_hist, Histogram)

        if other is None:
            other = _hist

        if memo is NOTHING:
            dict_copy = copy.copy(other.__dict__)
        else:
            dict_copy = copy.deepcopy(other.__dict__, memo)

        self._from_histogram_object(_hist, __dict__=dict_copy)

        for i, ax in enumerate(self.axes):
            if memo is NOTHING:
                new_metadata = copy.copy(ax._ax.raw_metadata)
            else:
                new_metadata = copy.deepcopy(ax._ax.raw_metadata, memo)
            # A growth axis wrapper holds a copy, so set the stored axis too.
            self._hist._set_axis_metadata(i, new_metadata)
            ax._ax.raw_metadata = new_metadata
            ax.__dict__ = new_metadata
        return self

    def _new_hist(self, _hist: CppHistogram, memo: Any = NOTHING) -> Self:
        """
        Return a new histogram given a new _hist, copying current metadata.
        """
        return self.__class__._clone(_hist, other=self, memo=memo)

    def _from_histogram_cpp(
        self, other: CppHistogram, *, __dict__: dict[str, Any]
    ) -> None:
        """
        Import a Cpp histogram.
        """
        self._variance_known = True
        self._hist = other
        self.__dict__.update(__dict__)
        self.axes = self._generate_axes_()

    def _from_histogram_object(
        self, other: Histogram[S], *, __dict__: dict[str, Any]
    ) -> None:
        """
        Convert self into a new histogram object based on another, possibly
        converting from a different subclass.
        """
        self._hist = other._hist
        self.__dict__ = copy.copy(other.__dict__)
        self.axes = self._generate_axes_()
        for ax in self.axes:
            # Give each axis its own metadata dict, so mutating it here does
            # not alias with the axes of `other` (or of each other).
            ax._ax.raw_metadata = copy.copy(ax._ax.raw_metadata)
            ax.__dict__ = ax._ax.raw_metadata
        self.__dict__.update(__dict__)

        # Allow custom behavior on either "from" or "to"
        other._export_bh_(self)
        self._import_bh_()

    def _import_bh_(self) -> None:
        """
        If any post-processing is needed to pass a histogram between libraries, a
        subclass can implement it here. self is the new instance in the current
        (converted-to) class.
        """

    @classmethod
    def _export_bh_(cls, self: Histogram[Any]) -> None:
        """
        If any preparation is needed to pass a histogram between libraries, a subclass can
        implement it here. cls is the current class being converted from, and self is the
        instance in the class being converted to.
        """

    def _generate_axes_(self) -> AxesTuple:
        """
        This is called to fill in the axes. Subclasses can override it if they need
        to change the axes tuple.
        """

        return AxesTuple(self._axis(i) for i in range(self.ndim))

    @property
    def _has_growth(self) -> bool:
        """
        True if any axis can grow. Growing invalidates views and axis objects.
        """
        return any(ax._ax.traits_growth for ax in self.axes)

    # Backward compat for metadata default
    def __getattr__(self, name: str) -> Any:
        if name == "metadata":
            msg = ".metadata was not set, returning None instead of Attribute error, boost-histogram 1.9+ will error."
            warnings.warn(msg, FutureWarning, stacklevel=2)
            return None
        return super().__getattribute__(name)

    def _to_uhi_(self) -> dict[str, Any]:
        """
        Convert to a UHI histogram.
        """
        return serialization.to_uhi(self)

    @classmethod
    def _from_uhi_(cls, inp: dict[str, Any], /) -> Self:
        """
        Convert from a UHI histogram.
        """
        return cls(serialization.from_uhi(inp))

    @property
    def ndim(self) -> int:
        """
        Number of axes (dimensions) of the histogram.
        """
        return self._hist.rank()

    @typing.overload
    def view(
        self: Histogram[bhs.Double]
        | Histogram[bhs.MultiCell]
        | Histogram[bhs.Unlimited],
        flow: bool = False,
    ) -> np.typing.NDArray[np.float64]: ...

    @typing.overload
    def view(
        self: Histogram[bhs.Int64] | Histogram[bhs.AtomicInt64],
        flow: bool = False,
    ) -> np.typing.NDArray[np.int64]: ...

    @typing.overload
    def view(self: Histogram[bhs.Weight], flow: bool = False) -> WeightedSumView: ...

    @typing.overload
    def view(self: Histogram[bhs.Mean], flow: bool = False) -> MeanView: ...

    @typing.overload
    def view(
        self: Histogram[bhs.WeightedMean], flow: bool = False
    ) -> WeightedMeanView: ...

    @typing.overload
    def view(
        self: Histogram[Any], flow: bool = False
    ) -> (
        np.typing.NDArray[np.float64]
        | np.typing.NDArray[np.int64]
        | WeightedSumView
        | WeightedMeanView
        | MeanView
    ): ...

    def view(
        self, flow: bool = False
    ) -> (
        np.typing.NDArray[np.float64]
        | np.typing.NDArray[np.int64]
        | WeightedSumView
        | WeightedMeanView
        | MeanView
    ):
        """
        Return a view into the data, optionally with overflow turned on.

        The view shares memory with the histogram. If the histogram has a
        growth axis, a fill that grows an axis moves the data, and the view
        then points at freed memory. Take a copy before such a fill.
        """
        return _to_view(self._hist.view(flow))

    def __array__(
        self,
        dtype: np.typing.DTypeLike | None = None,
        *,
        # pylint: disable-next=redefined-outer-name
        copy: bool | None = None,
    ) -> np.typing.NDArray[Any]:
        # The copy kw is new in NumPy 2.0
        kwargs = {}
        if copy is not None:
            kwargs["copy"] = copy
        return np.asarray(self.view(False), dtype=dtype, **kwargs)  # type: ignore[call-overload, no-any-return]

    __hash__ = None  # type: ignore[assignment]

    def allclose(
        self,
        other: object,
        *,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
        flow: bool = True,
        metadata: bool = False,
    ) -> bool:
        """
        Check whether two histograms are close to each other.

        Parameters
        ----------
        other : Histogram
            The histogram to compare against.
        rtol : float = 1e-05
            Relative tolerance for comparing edges and bins.
        atol : float = 1e-08
            Absolute tolerance for comparing edges and bins.
        equal_nan : bool = False
            Whether to compare NaNs as equal.
        flow : bool = True
            Whether to include underflow and overflow bins in the comparison.
        metadata : bool = False
            Whether to compare histogram and axis metadata.

        Returns
        -------
        bool
            True if the histograms are close, False otherwise.
        """
        if not isinstance(other, Histogram):
            return False

        if self.ndim != other.ndim:
            return False

        if self.storage_type != other.storage_type:
            return False

        if metadata and self.__dict__ != other.__dict__:
            return False

        for i in range(self.ndim):
            ax1 = self.axes[i]
            ax2 = other.axes[i]

            if ax1.size != ax2.size:
                return False

            if metadata and ax1.__dict__ != ax2.__dict__:
                return False

            if ax1.traits.continuous != ax2.traits.continuous:
                return False

            if ax1.traits.continuous:
                if not np.allclose(
                    ax1.edges,
                    ax2.edges,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=equal_nan,
                ):
                    return False
            else:
                if ax1.traits.ordered != ax2.traits.ordered:
                    return False
                if list(ax1) != list(ax2):
                    return False

        v1 = self.view(flow=flow)
        v2 = other.view(flow=flow)

        if v1.shape != v2.shape:
            return False

        if v1.dtype.names:
            for name in v1.dtype.names:
                if not np.allclose(
                    v1[name],  # type: ignore[index]
                    v2[name],  # type: ignore[index]
                    rtol=rtol,
                    atol=atol,
                    equal_nan=equal_nan,
                ):
                    return False
        elif not np.allclose(
            v1,
            v2,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        ):
            return False

        return True

    def __eq__(self, other: object) -> bool:
        return hasattr(other, "_hist") and self._hist == other._hist

    def __add__(self, other: Histogram[S] | np.typing.NDArray[Any] | float) -> Self:
        result = self.copy(deep=False)
        return result.__iadd__(other)

    def __iadd__(self, other: Histogram[S] | np.typing.NDArray[Any] | float) -> Self:
        if isinstance(other, (int, float)) and other == 0:
            return self
        self._compute_inplace_op("__iadd__", other)

        # Addition may change the axes if they can grow
        self.axes = self._generate_axes_()

        return self

    def __radd__(self, other: np.typing.NDArray[Any] | float) -> Self:
        return self + other

    def __sub__(self, other: Histogram[S] | np.typing.NDArray[Any] | float) -> Self:
        result = self.copy(deep=False)
        return result.__isub__(other)

    def __isub__(self, other: Histogram[S] | np.typing.NDArray[Any] | float) -> Self:
        if isinstance(other, (int, float)) and other == 0:
            return self
        self._compute_inplace_op("__isub__", other)

        self.axes = self._generate_axes_()

        return self

    def __rsub__(self, other: np.typing.NDArray[Any] | float) -> Self:
        # Subtraction is not commutative, so unlike __radd__ this cannot defer
        # to the forward operator: other - self == -(self - other).
        return (self - other) * -1

    # If these fail, the underlying object throws the correct error
    def __mul__(self, other: Histogram[S] | np.typing.NDArray[Any] | float) -> Self:
        result = self.copy(deep=False)
        return result.__imul__(other)

    def __rmul__(self, other: np.typing.NDArray[Any] | float) -> Self:
        return self * other

    def __truediv__(self, other: Histogram[S] | np.typing.NDArray[Any] | float) -> Self:
        result = self.copy(deep=False)
        return result.__itruediv__(other)

    def __itruediv__(
        self, other: Histogram[S] | np.typing.NDArray[Any] | float
    ) -> Self:
        # Division should produce floating-point results, so promote integer
        # storages to Double. Histogram/histogram division in C++ requires both
        # operands to share a storage, so the divisor is promoted to match.
        self._convert_int_storage_to_double()
        if isinstance(other, Histogram):
            other = self._as_double_cpp(other._hist)  # type: ignore[assignment]
        elif isinstance(other, _histogram_types):
            other = self._as_double_cpp(other)  # type: ignore[assignment]
        return self._compute_inplace_op("__itruediv__", other)

    def __rtruediv__(self, other: np.typing.NDArray[Any] | float) -> Self:
        # Division is not commutative, so unlike __rmul__ this cannot defer to
        # the forward operator: divide other by each cell. Promote integer
        # storages to Double first, matching __itruediv__.
        result = self.copy(deep=False)
        result._convert_int_storage_to_double()
        view = result.view(flow=True)
        # Empty bins divide to inf/nan; suppress the warnings as elsewhere.
        with np.errstate(divide="ignore", invalid="ignore"):
            np.true_divide(other, view, out=view)
        result._variance_known = False
        return result

    def __imul__(self, other: Histogram[S] | np.typing.NDArray[Any] | float) -> Self:
        # Multiplying by a non-integer scalar/array should promote an integer
        # storage to Double, matching __itruediv__, instead of leaking a
        # numpy UFuncTypeError from the in-place view multiply.
        if not isinstance(other, (Histogram, *_histogram_types)) and not np.issubdtype(
            np.asarray(other).dtype, np.integer
        ):
            self._convert_int_storage_to_double()
        return self._compute_inplace_op("__imul__", other)

    @staticmethod
    def _as_double_cpp(cpp_hist: CppHistogram) -> CppHistogram:
        """
        Return a Double-storage copy of an integer-storage (Int64/AtomicInt64)
        C++ histogram, so that division produces floating-point results instead
        of truncating. Returns the input unchanged for storages that are already
        floating point or richer.
        """
        if cpp_hist._storage_type not in {
            _core.storage.int64,
            _core.storage.atomic_int64,
        }:
            return cpp_hist

        cpp_axes = [cpp_hist.axis(i) for i in range(cpp_hist.rank())]
        new_hist = _core.hist.any_double(cpp_axes, _core.storage.double())
        new_hist.view(flow=True)[...] = cpp_hist.view(flow=True)
        return new_hist

    def _convert_int_storage_to_double(self) -> None:
        """
        Convert an integer storage to Double in place (see _as_double_cpp).
        """
        new_hist = self._as_double_cpp(self._hist)
        if new_hist is not self._hist:
            self._hist = new_hist
            # The axes belong to the old histogram; point them at the new one.
            self.axes = self._generate_axes_()

    def _hist_inplace_op(self, name: str, other: CppHistogram) -> None:
        if self._hist._storage_type is not other._storage_type:
            symbol = _INPLACE_OP_SYMBOLS.get(name, name)
            other_storage = typing.cast(
                type[Storage], cast(self, other._storage_type, Storage)
            )
            msg = (
                f"Cannot {symbol} histograms with different storage types: "
                f"{self.storage_type.__name__} and {other_storage.__name__}"
            )
            raise TypeError(msg)

        # The underlying C++ histogram only exposes the in-place dunders its
        # storage supports (e.g. weight/mean/multi_cell storages have no
        # __isub__). Calling a missing one raises a confusing AttributeError
        # leaking the dunder name, so surface a clear error instead.
        op = getattr(self._hist, name, None)
        if op is None:
            symbol = _INPLACE_OP_SYMBOLS.get(name, name)
            msg = (
                f"The {self.storage_type.__name__} storage does not support the "
                f"{symbol!r} operation between two histograms"
            )
            raise TypeError(msg)
        op(other)

    def _compute_inplace_op(
        self, name: str, other: Histogram[S] | np.typing.NDArray[Any] | float
    ) -> Self:
        # Also takes CppHistogram, but that confuses mypy because it's hard to pick out
        if isinstance(other, Histogram):
            self._hist_inplace_op(name, other._hist)
        elif isinstance(other, _histogram_types):
            self._hist_inplace_op(name, other)
        elif hasattr(other, "shape") and other.shape:
            assert not isinstance(other, float)

            if len(other.shape) != self.ndim:
                msg = f"Number of dimensions {len(other.shape)} must match histogram {self.ndim}"
                raise ValueError(msg)

            if all(a in {b, 1} for a, b in zip(other.shape, self.shape, strict=False)):
                view = self.view(flow=False)
                getattr(view, name)(other)
            elif all(
                a in {b, 1} for a, b in zip(other.shape, self.axes.extent, strict=False)
            ):
                view = self.view(flow=True)
                getattr(view, name)(other)
            else:
                msg = f"Wrong shape {other.shape}, expected {self.shape} or {self.axes.extent}"
                raise ValueError(msg)
        else:
            view = self.view(flow=True)
            getattr(view, name)(other)

        self._variance_known = False
        return self

    # TODO: Marked as too complex by flake8. Should be factored out a bit.
    def fill(
        self,
        *args: ArrayLike | str,
        weight: ArrayLike | None = None,
        sample: ArrayLike | None = None,
        threads: int | None = None,
    ) -> Self:
        """
        Insert data into the histogram.

        Parameters
        ----------
        *args : Union[Array[float], Array[int], Array[str], float, int, str]
            Provide one value or array per dimension.
        weight : list[Union[Array[float], Array[int], float, int, str]]]
            Provide weights (only if the histogram storage supports it)
        sample : list[Union[Array[float], Array[int], Array[str], float, int, str]]]
            Provide samples (only if the histogram storage supports it)
        threads : Optional[int]
            Fill with threads. Defaults to None, which does not activate
            threaded filling.  Using 0 will automatically pick the number of
            available threads (usually two per core). A continuous growth axis
            (such as a growing Regular axis) cannot be merged across threads
            yet, so such a fill raises instead.
        """

        if self._hist._storage_type is _core.storage.mean:
            mean_storage_sample_check(sample)

        if (
            self._hist._storage_type
            not in {
                _core.storage.weight,
                _core.storage.mean,
                _core.storage.weighted_mean,
            }
            and weight is not None
        ):
            self._variance_known = False

        if self._hist._storage_type is _core.storage.multi_cell:
            # Use weight keyword for MultiCell filling even though it uses sample on the C++ backend
            sample = weight
            weight = None

        # Convert to NumPy arrays
        args_ars = _fill_cast(args)
        weight_ars = _fill_cast(weight)
        sample_ars = _fill_cast(sample)

        # Broadcast scalar positional args to match sample length when sample is an array.
        # This allows e.g. h.fill(0, sample=[1, 2, 3]) to work for Mean/WeightedMean storage.
        if sample_ars is not None:
            sample_arr = np.asarray(sample_ars)
            if sample_arr.ndim > 0:
                sample_len = len(sample_arr)
                if sample_len > 1:
                    args_ars = tuple(
                        np.full(sample_len, a) if np.ndim(a) == 0 else a
                        for a in args_ars
                    )

        if threads == 0:
            threads = cpu_count()

        growth = self._has_growth

        if threads is None or threads == 1:
            self._hist.fill(*args_ars, weight=weight_ars, sample=sample_ars)
            if growth:
                # A growing fill replaces the axes.
                self.axes = self._generate_axes_()
            return self

        if self._hist._storage_type in {
            _core.storage.mean,
            _core.storage.weighted_mean,
        }:
            raise RuntimeError("Mean histograms do not support threaded filling")

        # If everything is scalar, there is only a single fill; threading would
        # incorrectly repeat it, so fill directly instead.
        if (
            all(isinstance(a, str) or np.ndim(a) == 0 for a in args_ars)
            and (weight_ars is None or np.ndim(weight_ars) == 0)
            and (sample_ars is None or np.ndim(sample_ars) == 0)
        ):
            self._hist.fill(*args_ars, weight=weight_ars, sample=sample_ars)
            if growth:
                self.axes = self._generate_axes_()
            return self

        data: list[list[Any]] = []
        for a in args_ars:
            if isinstance(a, str) or np.ndim(a) == 0:
                # Scalars broadcast against each thread's chunk
                data.append([a] * threads)
            else:
                data.append(list(np.array_split(np.asarray(a), threads)))

        weights: list[Any]
        if weight_ars is None or np.ndim(weight_ars) == 0:
            assert threads is not None
            weights = [weight_ars] * threads
        else:
            weights = np.array_split(np.asarray(weight_ars), threads)

        samples: list[Any]
        if sample_ars is None or np.ndim(sample_ars) == 0:
            assert threads is not None
            samples = [sample_ars] * threads
        else:
            samples = np.array_split(np.asarray(sample_ars), threads)

        if self._hist._storage_type is _core.storage.atomic_int64:

            def work(
                weight: ArrayLike | None,
                sample: ArrayLike | None,
                *args: np.typing.NDArray[Any],
            ) -> None:
                self._hist.fill(*args, weight=weight, sample=sample)

        else:
            sum_lock = threading.Lock()

            def work(
                weight: ArrayLike | None,
                sample: ArrayLike | None,
                *args: np.typing.NDArray[Any],
            ) -> None:
                # The copy and the merge both release the GIL, so a copy made
                # while another worker merges would read a histogram that is
                # changing. The lock keeps every access to self._hist ordered.
                with sum_lock:
                    local_hist = copy.copy(self._hist)
                local_hist.reset()
                local_hist.fill(*args, weight=weight, sample=sample)
                with sum_lock:
                    self._hist += local_hist

        # A worker exception must not be lost; the histogram would silently
        # hold too few entries.
        errors: list[Exception] = []
        error_lock = threading.Lock()

        def fun(
            weight: ArrayLike | None,
            sample: ArrayLike | None,
            *args: np.typing.NDArray[Any],
        ) -> None:
            try:
                work(weight, sample, *args)
            # Any worker error must reach the caller, so catch them all
            except Exception as err:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                with error_lock:
                    errors.append(err)

        thread_list = [
            threading.Thread(target=fun, args=arrays)
            for arrays in zip(weights, samples, *data, strict=False)
        ]

        for thread in thread_list:
            thread.start()

        for thread in thread_list:
            thread.join()

        if errors:
            raise errors[0]

        if growth:
            # The merge of the per-thread copies replaces the axes.
            self.axes = self._generate_axes_()

        return self

    def __str__(self) -> str:
        """
        A rendering of the histogram is made using ASCII or unicode characters
        (whatever is supported by the terminal). What exactly is displayed is
        still experimental. Do not rely on any particular rendering.
        """
        # TODO check the terminal width and adjust the presentation
        # only use for 1D, fall back to repr for ND
        if self._hist.rank() != 1:
            return repr(self)
        s = str(self._hist)
        # get rid of first line and last character
        return s[s.index("\n") + 1 : -1]

    def _axis(self, i: int = 0) -> Axis:
        """
        Get N-th axis.
        """
        return cast(self, self._hist.axis(i), Axis)

    @property
    def storage_type(self) -> type[S]:
        return cast(self, self._hist._storage_type, Storage)  # type: ignore[return-value]

    @property
    def _storage_type(self) -> type[S]:
        warnings.warn(
            "Accessing storage type has changed from _storage_type to storage_type, and will be removed in future.",
            FutureWarning,
            stacklevel=2,
        )
        return cast(self, self._hist._storage_type, Storage)  # type: ignore[return-value]

    @property
    def storage(self) -> S:
        """
        New storage matching the one the histogram was constructed with.
        """
        if issubclass(self.storage_type, bhs.MultiCell):
            return self.storage_type(self._hist.nelem())  # type: ignore[attr-defined]
        return self.storage_type()

    def _reduce(self, *args: Any) -> Self:
        return self._new_hist(self._hist.reduce(*args))

    def __copy__(self) -> Self:
        return self._new_hist(copy.copy(self._hist))

    def __deepcopy__(self, memo: Any) -> Self:
        return self._new_hist(copy.deepcopy(self._hist), memo=memo)

    def __getstate__(self) -> tuple[int, dict[str, Any]]:
        """
        Version 0.8: metadata added
        Version 0.11: version added and set to 0. metadata/_hist replaced with dict.
        Version 0.12: _variance_known is now in the dict (no format change)

        ``dict`` contains __dict__ with added "_hist"
        """
        local_dict = copy.copy(self.__dict__)
        local_dict["_hist"] = self._hist
        # Version 0 of boost-histogram pickle state
        return (0, local_dict)

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, tuple):
            if state[0] == 0:
                for key, value in state[1].items():
                    setattr(self, key, value)

                # Added in 0.12
                if "_variance_known" not in state[1]:
                    self._variance_known = True
            else:
                msg = f"Cannot open boost-histogram pickle v{state[0]}"
                raise RuntimeError(msg)

        else:  # Classic (0.10 and before) state
            self._hist = state["_hist"]
            self._variance_known = True
            self.metadata = state.get("metadata", None)
            for i in range(self._hist.rank()):
                self._hist._set_axis_metadata(
                    i, {"metadata": self._hist.axis(i).raw_metadata}
                )

        self.axes = self._generate_axes_()

    def __repr__(self) -> str:
        newline = "\n  "
        first_newline = newline if len(self.axes) > 1 else ""
        storage_newline = (
            newline if len(self.axes) > 1 else " " if len(self.axes) > 0 else ""
        )
        sep = "," if len(self.axes) > 0 else ""
        ret = f"{self.__class__.__name__}({first_newline}"
        ret += f",{newline}".join(repr(ax) for ax in self.axes)
        ret += f"{sep}{storage_newline}storage={self.storage}"
        ret += ")"
        outer = self.sum(flow=True)
        # Accumulators (Mean, WeightedSum, ...) have no __bool__, so they are
        # always truthy; compare against a fresh instance to detect "empty".
        if isinstance(outer, (int, float)):
            non_empty = bool(outer)
        else:
            non_empty = outer != type(outer)()
        if non_empty:
            inner = self.sum(flow=False)
            ret += f" # Sum: {inner}"
            if inner != outer:
                ret += f" ({outer} with flow)"
        return ret

    def _compute_uhi_index(self, index: InnerIndexing, axis: int) -> SimpleIndexing:
        """
        Converts an expression that contains UHI locators to one that does not.
        """
        # Support sum and rebin directly
        if index is sum or hasattr(index, "factor"):  # type: ignore[comparison-overlap,redundant-expr]
            return slice(None, None, index)

        # General locators
        # Note that MyPy doesn't like these very much - the fix
        # will be to properly set input types
        if callable(index):
            return index(self.axes[axis])

        # NumPy integer arrays pass through untouched; they trigger vectorized
        # gather/scatter in __getitem__/__setitem__ rather than per-element access.
        if isinstance(index, np.ndarray):
            return index

        if isinstance(index, float):
            raise TypeError(f"Index {index} must be an integer, not float")

        if isinstance(index, SupportsIndex):
            idx = int(index)
            size: int = self._hist.axis(axis).size
            if not -size <= idx < size:
                raise IndexError("histogram index is out of range")
            return idx % size

        return index

    def _compute_commonindex(
        self, index: IndexingExpr
    ) -> list[SupportsIndex | slice | Mapping[int, SupportsIndex | slice]]:
        """
        Takes indices and returns two iterables; one is a tuple or dict of the
        original, Ellipsis expanded index, and the other returns index,
        operation value pairs.
        """
        indexes: list[Any]

        # Shorten the computations with direct access to raw object
        hist = self._hist

        # Support dict access
        if hasattr(index, "items"):
            indexes = [slice(None)] * hist.rank()
            for k, v in index.items():
                indexes[k] = v

        # Normalize -> h[i] == h[i,]
        else:
            tuple_index = (index,) if not isinstance(index, tuple) else index

            # Now a list
            indexes = _expand_ellipsis(tuple_index, hist.rank())

        if len(indexes) != hist.rank():
            raise IndexError("Wrong number of indices for histogram")

        # Allow [bh.loc(...)] to work
        # TODO: could be nicer making a new list via a comprehension
        for i in range(len(indexes)):  # pylint: disable=consider-using-enumerate
            # Support list of UHI indexers
            if isinstance(indexes[i], list):
                indexes[i] = [self._compute_uhi_index(ind, i) for ind in indexes[i]]
            else:
                indexes[i] = self._compute_uhi_index(indexes[i], i)

        return indexes

    def _flow_pick_index(self, axis: int, idx: int) -> int:
        """
        Map a UHI-resolved scalar index for ``axis`` to an offset into the
        ``flow=True`` view. Regular indices are 0..size-1; the sentinels -1
        and size (produced by ``bh.underflow``/``bh.overflow``/``bh.loc`` on
        an out-of-range value) select the flow bins, raising IndexError if
        the axis does not have the requested flow bin.
        """
        traits = self.axes[axis].traits
        size = self._hist.axis(axis).size
        offset = 1 if traits.underflow else 0
        if idx == -1:
            if not traits.underflow:
                raise IndexError(f"Axis {axis} has no underflow bin")
            return 0
        if idx == size:
            if not traits.overflow:
                raise IndexError(f"Axis {axis} has no overflow bin")
            return size + offset
        return idx + offset

    @staticmethod
    def _flow_slice_bound(v: int | None, default: int, size: int, offset: int) -> int:
        """
        Resolve one bound of a plain integer slice used in vectorized
        indexing: None and negative values are resolved the same way NumPy
        would on the flow=False view, then shifted onto the flow=True view.
        """
        if v is None:
            v = default
        elif v < 0:
            v += size
        return v + offset

    def _compute_vectorized_index(self, indexes: list[Any]) -> tuple[Any, ...]:
        """
        Build a NumPy fancy-index tuple from already-normalized indexes for
        vectorized cell access (gather/scatter) into the ``flow=True`` view.
        Each axis may be an integer array, an integer (including the
        ``bh.underflow``/``bh.overflow`` sentinels), or a plain integer
        slice. Rebin/sum/locator slices and categorical lists are rejected
        with a pointer to ``.view()``.
        """
        view_index: list[Any] = []
        for i, ind in enumerate(indexes):
            offset = 1 if self.axes[i].traits.underflow else 0
            if isinstance(ind, np.ndarray):
                view_index.append(ind + offset)
            elif isinstance(ind, slice):
                if ind.step is not None or not all(
                    s is None or isinstance(s, int) for s in (ind.start, ind.stop)
                ):
                    msg = (
                        f"Vectorized (array) indexing on axis {i} only supports plain "
                        "integer slices; use .view() for rebin, sum, or locator slices"
                    )
                    raise IndexError(msg)
                # Bounds are plain non-flow indices (matching scalar/array
                # indexing semantics: no flow bins unless asked for), with
                # None and negative values resolved the same way as NumPy
                # would on the flow=False view, then shifted onto the
                # flow=True view.
                size = self._hist.axis(i).size
                start = self._flow_slice_bound(ind.start, 0, size, offset)
                stop = self._flow_slice_bound(ind.stop, size, size, offset)
                view_index.append(slice(start, stop))
            elif isinstance(ind, SupportsIndex):
                view_index.append(self._flow_pick_index(i, ind.__index__()))
            else:
                msg = (
                    "Vectorized (array) indexing only supports integer arrays, "
                    f"integers, and integer slices; got {type(ind).__name__} on axis {i}"
                )
                raise IndexError(msg)

        if isinstance(self._hist, _core.hist.any_multi_cell):
            # The buffer of a MultiCell histogram has the cell index as its first
            # dimension, which is not part of the user-facing axis indexing.
            view_index.insert(0, slice(None, None, None))

        return tuple(view_index)

    @typing.overload
    def to_numpy(
        self, flow: bool = ..., *, dd: Literal[False] = ..., view: bool = ...
    ) -> tuple[np.typing.NDArray[Any], ...]: ...

    @typing.overload
    def to_numpy(
        self, flow: bool = ..., *, dd: Literal[True], view: bool = ...
    ) -> tuple[np.typing.NDArray[Any], tuple[np.typing.NDArray[np.float64], ...]]: ...

    @typing.overload
    def to_numpy(
        self, flow: bool = ..., *, dd: bool, view: bool = ...
    ) -> (
        tuple[np.typing.NDArray[Any], ...]
        | tuple[np.typing.NDArray[Any], tuple[np.typing.NDArray[np.float64], ...]]
    ): ...

    def to_numpy(
        self, flow: bool = False, *, dd: bool = False, view: bool = False
    ) -> (
        tuple[np.typing.NDArray[Any], ...]
        | tuple[np.typing.NDArray[Any], tuple[np.typing.NDArray[np.float64], ...]]
    ):
        """
        Convert to a NumPy style tuple of return arrays. Edges are converted to
        match NumPy standards, with upper edge inclusive, unlike
        boost-histogram, where upper edge is exclusive.

        Parameters
        ----------
        flow : bool = False
            Include the flow bins.
        dd : bool = False
            Use the histogramdd return syntax, where the edges are in a tuple.
            Otherwise, this is the histogram/histogram2d return style.
        view : bool  = False
            The behavior for the return value. By default, this will return
            array of the values only regardless of the storage (which is all
            NumPy's histogram function can do). view=True will return the
            boost-histogram view of the storage.

        Return
        ------
        contents : Array[Any]
            The bin contents
        *edges : Array[float]
            The edges for each dimension
        """

        hist = self.view(flow=flow) if view else self.values(flow=flow)
        # Compute the edges directly; this avoids the deep copy of the bin
        # contents that the C++ ``to_numpy`` helper would make.
        edges = [
            _numpy_compatible_edges(self._hist.axis(i), flow) for i in range(self.ndim)
        ]

        return (hist, edges) if dd else (hist, *edges)  # type: ignore[return-value]

    def copy(self, *, deep: bool = True) -> Self:
        """
        Make a copy of the histogram. Defaults to making a
        deep copy (axis metadata copied); use deep=False
        to avoid making a copy of axis metadata.
        """

        return copy.deepcopy(self) if deep else copy.copy(self)

    def reset(self) -> Self:
        """
        Clear the bin counters.
        """
        self._hist.reset()
        return self

    def empty(self, flow: bool = False) -> bool:
        """
        Check to see if the histogram has any non-default values.
        You can use flow=True to check flow bins too.
        """
        return self._hist.empty(flow)

    @typing.overload
    def sum(
        self: Histogram[bhs.Double]
        | Histogram[bhs.Int64]
        | Histogram[bhs.AtomicInt64]
        | Histogram[bhs.Unlimited],
        flow: bool = False,
    ) -> float: ...

    @typing.overload
    def sum(self: Histogram[bhs.MultiCell], flow: bool = False) -> list[float]: ...

    @typing.overload
    def sum(self: Histogram[bhs.Weight], flow: bool = False) -> WeightedSum: ...

    @typing.overload
    def sum(self: Histogram[bhs.Mean], flow: bool = False) -> Mean: ...

    @typing.overload
    def sum(self: Histogram[bhs.WeightedMean], flow: bool = False) -> WeightedMean: ...

    @typing.overload
    def sum(self, flow: bool = False) -> float | Accumulator | list[float]: ...

    def sum(self, flow: bool = False) -> float | Accumulator | list[float]:
        """
        Compute the sum over the histogram bins (optionally including the flow bins).
        """
        return self._hist.sum(flow)  # type: ignore[no-any-return]

    @property
    def size(self) -> int:
        """
        Total number of bins in the histogram (including underflow/overflow).
        """
        return self._hist.size()

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Tuple of axis sizes (not including underflow/overflow).
        """
        return self.axes.size

    @typing.overload
    def __getitem__(self: FloatHists, index: IndexingExpr) -> FloatHists | float: ...

    @typing.overload
    def __getitem__(self: IntHists, index: IndexingExpr) -> IntHists | int: ...

    @typing.overload
    def __getitem__(
        self: ListHists, index: IndexingExpr
    ) -> ListHists | list[float]: ...

    @typing.overload
    def __getitem__(
        self: WeightHists, index: IndexingExpr
    ) -> WeightHists | WeightedSum: ...

    @typing.overload
    def __getitem__(self: MeanHists, index: IndexingExpr) -> MeanHists | Mean: ...

    @typing.overload
    def __getitem__(
        self: WeightedMeanHists, index: IndexingExpr
    ) -> WeightedMeanHists | WeightedMean: ...

    @typing.overload
    def __getitem__(
        self, index: IndexingExpr
    ) -> Self | float | list[float] | int | Accumulator: ...

    def __getitem__(
        self, index: IndexingExpr
    ) -> Self | float | Accumulator | list[float] | int | np.typing.NDArray[Any]:
        indexes = self._compute_commonindex(index)

        # Vectorized (NumPy array) indexing gathers scattered cells through the
        # buffer instead of building a new histogram. Only ndarray indices
        # trigger this; lists keep their categorical pick semantics.
        if any(isinstance(a, np.ndarray) for a in indexes):
            return self.view(flow=True)[self._compute_vectorized_index(indexes)]

        # Early return for all-integer case
        if all(isinstance(a, SupportsIndex) for a in indexes):
            return self._hist.at(*indexes)  # type: ignore[no-any-return, arg-type]

        integrations = set[int]()
        slices = list[_core.algorithm.reduce_command]()
        pick_each = dict[int, int]()
        pick_set = dict[int, list[int]]()
        reduced: CppHistogram | None = None

        for i, ind in enumerate(indexes):
            match ind:
                case SupportsIndex():
                    pick_each[i] = self._flow_pick_index(i, ind.__index__())
                # str/bytes are Sequences but not valid indices; they fall
                # through to the IndexError below.
                case collections.abc.Sequence() if not isinstance(ind, (str, bytes)):  # type: ignore[unreachable]
                    pick_set[i] = list(ind)
                case slice(start=start, stop=stop, step=step):
                    reduced, new_slices, new_integrations = self._handle_slice(
                        i, start, stop, step, reduced
                    )
                    slices.extend(new_slices)
                    integrations.update(new_integrations)
                case _:
                    raise IndexError(
                        "Must be a slice, an integer, or follow the locator protocol."
                    )

        if (slices or pick_set or pick_each or integrations) and not reduced:
            reduced = self._hist
        elif not reduced:
            reduced = copy.copy(self._hist)

        if pick_each:
            tuple_slice = tuple(
                pick_each.get(i, slice(None)) for i in range(reduced.rank())
            )

            if isinstance(self._hist, _core.hist.any_multi_cell):
                # View of multi cell histograms has as first (index 0) dimension the cell index
                # Add a full slice to the beginning of the slicing expression to adept for this cell index
                # e.g. a slice like [0, :, 3] is converted to [:, 0, :, 3]
                tuple_slice = (slice(None, None, None), *tuple_slice)

            logger.debug("Slices for pick each: %s", tuple_slice)
            axes = [
                reduced.axis(i) for i in range(reduced.rank()) if i not in pick_each
            ]
            logger.debug("Axes: %s", axes)
            new_reduced: _core.hist._BaseHistogram | _core.hist.any_multi_cell = (
                reduced.__class__(axes)
            )
            if isinstance(reduced, _core.hist.any_multi_cell) and isinstance(
                new_reduced, _core.hist.any_multi_cell
            ):
                # The constructor in reduced.__class__(axes) does not take care of the number of cells.
                # If reduced is a multi cell histogram, we have to set the number of cells per bin manually for new_reduced
                new_reduced.reset_nelem(reduced.nelem())
            new_reduced.view(flow=True)[...] = reduced.view(flow=True)[tuple_slice]
            reduced = new_reduced
            integrations = {i - sum(j <= i for j in pick_each) for i in integrations}
            pick_set = {
                i - sum(j <= i for j in pick_each): v for i, v in pick_set.items()
            }
            for slice_ in slices:
                slice_.iaxis -= sum(j <= slice_.iaxis for j in pick_each)

        if slices:
            logger.debug("Reduce with %s", slices)
            reduced = reduced.reduce(*slices)

        if pick_set:
            warnings.warn(
                "List indexing selection is experimental. Removed bins are not placed in overflow.",
                stacklevel=2,
            )
            logger.debug("Slices for picking sets: %s", pick_set)
            axes = [reduced.axis(i) for i in range(reduced.rank())]
            reduced_view = reduced.view(flow=True)
            for i in pick_set:
                selection = copy.copy(pick_set[i])
                ax = reduced.axis(i)
                if ax.traits_ordered:
                    msg = f"Axis {i} is not a categorical axis, cannot pick with list: {ax}"
                    raise RuntimeError(msg)

                if ax.traits_overflow and ax.size not in pick_set[i]:
                    selection.append(ax.size)

                new_axis = axes[i].__class__([axes[i].value(j) for j in pick_set[i]])  # type: ignore[call-arg]
                new_axis.raw_metadata = axes[i].raw_metadata
                axes[i] = new_axis
                reduced_view = np.take(reduced_view, selection, axis=i)

            logger.debug("Axes: %s", axes)
            new_reduced = reduced.__class__(axes)
            new_reduced.view(flow=True)[...] = reduced_view
            reduced = new_reduced

        if integrations:
            projections = [i for i in range(reduced.rank()) if i not in integrations]
            reduced = reduced.project(*projections)

        return self._new_hist(reduced) if reduced.rank() > 0 else reduced.sum(flow=True)

    @staticmethod
    def _empty_slice_msg(i: int, start: Any, stop: Any) -> str:
        return (
            f"Slice [{start}:{stop}] on axis {i} selects no bins; boost-histogram "
            "axes cannot have zero bins (NumPy would return an empty array here)"
        )

    def _handle_slice(
        self,
        i: int,
        start: int | None,
        stop: int | None,
        step: int
        | slice
        | Mapping[int, SupportsIndex | slice]
        | Callable[[Any], int]
        | None,
        reduced: CppHistogram | None,
    ) -> tuple[CppHistogram | None, list[_core.algorithm.reduce_command], set[int]]:
        if any(isinstance(v, slice) for v in (start, stop, step)):
            msg = (
                "You have put a slice in a slice. Did you forget curly braces [{...}]?"
            )
            raise TypeError(msg)

        slices = list[_core.algorithm.reduce_command]()
        integrations = set[int]()

        if start is None and stop is None and step is None:
            return reduced, slices, integrations

        start_int, stop_int = self.axes[i]._process_loc(start, stop)
        groups = []
        new_axis = None
        merge = 1
        has_bounds = start is not None or stop is not None
        match step:
            case x if x is sum:  # https://github.com/oracle/graalpython/issues/620
                integrations.add(i)
                if has_bounds:
                    if start_int >= stop_int:
                        raise IndexError(self._empty_slice_msg(i, start, stop))
                    slices.append(
                        _core.algorithm.slice(
                            i, start_int, stop_int, _core.algorithm.slice_mode.crop
                        )
                    )
                return reduced, slices, integrations
            case None:
                pass
            case object(factor=x) if x is not None:
                merge = x
            case object(axis_mapping=x) if x is not None:
                # Groups are applied against the sliced (start:stop) range,
                # not the full axis, so crop to that range first.
                axis_for_groups = self.axes[i]
                if has_bounds:
                    if start_int >= stop_int:
                        raise IndexError(self._empty_slice_msg(i, start, stop))
                    reduced = (reduced or self._hist).reduce(
                        _core.algorithm.slice(
                            i, start_int, stop_int, _core.algorithm.slice_mode.crop
                        )
                    )
                    axis_for_groups = cast(self, reduced.axis(i), Axis)
                tmp_both = x(axis_for_groups)
                if tmp_both is None:
                    msg = "The third argument to a slice must be rebin or projection"
                    raise IndexError(msg)
                groups, new_axis = tmp_both
                if new_axis is None and not axis_for_groups.traits.ordered:
                    msg = (
                        f"Group rebin on categorical axis {i} needs an explicit "
                        "axis= (the merged category axis); it cannot be inferred "
                        "automatically"
                    )
                    raise ValueError(msg)
            case x if callable(x):
                raise NotImplementedError
            case _:
                msg = "The third argument to a slice must be rebin or projection"
                raise IndexError(msg)

        assert isinstance(start_int, int)
        assert isinstance(stop_int, int)
        # rebinning with factor
        if len(groups) == 0:
            # NumPy returns an empty array for slices like [1:-15] or [5:2];
            # a Boost.Histogram axis cannot have zero bins, so refuse with a
            # clear message instead of the low-level "begin < end required".
            if min(stop_int, self.axes[i].size) <= max(start_int, 0):
                raise IndexError(self._empty_slice_msg(i, start, stop))
            slices.append(
                _core.algorithm.slice_and_rebin(i, start_int, stop_int, merge)
            )
        # rebinning with groups
        else:
            reduced = self._rebin_with_groups(
                reduced or self._hist, i, groups, new_axis
            )
        return reduced, slices, integrations

    def _rebin_with_groups(
        self, reduced: CppHistogram, i: int, groups: list[int], new_axis: Any
    ) -> CppHistogram:
        """Handle rebinning with groups."""
        axes = [reduced.axis(x) for x in range(reduced.rank())]
        reduced_view = reduced.view(flow=True)
        new_axes_indices = [axes[i].edges[0]]

        j = 0
        for group in groups:
            new_axes_indices += [axes[i].edges[j + group]]
            j += group

        if new_axis is None:
            new_axis = Variable(
                new_axes_indices,
                __dict__=axes[i].raw_metadata,
                underflow=axes[i].traits_underflow,
                overflow=axes[i].traits_overflow,
            )
        old_axis = axes[i]
        axes[i] = new_axis._ax

        logger.debug("Axes: %s", axes)

        new_reduced: _core.hist._BaseHistogram | _core.hist.any_multi_cell
        new_reduced = reduced.__class__(axes)
        if isinstance(reduced, _core.hist.any_multi_cell) and isinstance(
            new_reduced, _core.hist.any_multi_cell
        ):
            # The constructor in reduced.__class__(axes) does not take care of the number of cells.
            # If reduced is a multi cell histogram, we have to set the number of cells per bin manually for new_reduced
            new_reduced.reset_nelem(reduced.nelem())
        new_view = new_reduced.view(flow=True)

        # Views of multi cell histograms have the cell index as the first
        # (index 0) dimension, so the axis position within the view is
        # shifted by one.
        view_i = i + 1 if isinstance(reduced, _core.hist.any_multi_cell) else i

        groups = list(groups)  # do not modify the caller's list
        j = 0
        new_j_base = 0

        if old_axis.traits_underflow and axes[i].traits_underflow:
            groups.insert(0, 1)
        elif axes[i].traits_underflow:
            new_j_base = 1
        elif old_axis.traits_underflow:
            # The new axis has no underflow bin: skip the old underflow bin
            # here. For unordered (categorical) axes its contents are folded
            # into the new overflow bin below; otherwise they are dropped.
            j = 1

        if old_axis.traits_overflow and axes[i].traits_overflow:
            groups.append(1)
        # If the old axis has an overflow bin but the new one does not, the
        # old overflow contents are dropped (the bin is simply not consumed).

        for new_j, group in enumerate(groups):
            for _ in range(group):
                _combine_group_contents(
                    new_view, reduced_view, view_i, j, new_j + new_j_base
                )
                j += 1

        if (
            old_axis.traits_underflow
            and not axes[i].traits_underflow
            and not axes[i].traits_ordered
            and axes[i].traits_overflow
        ):
            # On an unordered (categorical) axis every out-of-range entry
            # lands in the overflow bin, so the old underflow contents are
            # added to the new overflow bin -- exactly once.
            _combine_group_contents(new_view, reduced_view, view_i, 0, -1)

        return new_reduced

    def __setitem__(self, index: IndexingExpr, value: ArrayLike | Accumulator) -> None:
        """
        There are several supported possibilities:

            h[slice] = array # same size

        If an array is given to a compatible slice, it is set.

            h[a:] = array # One larger

        If an array is given that does not match, if it does match the
        with-overflow size, it fills that.

            h[a:] = h2

        If another histogram is given, that must either match with or without
        overflow, where the overflow bins must be overflow bins (that is,
        you cannot set a histogram's flow bins from another histogram that
        is 2 larger). If you don't want this level of type safety, just use
        ``h[...] = h2.view()``.
        """
        indexes = self._compute_commonindex(index)

        # Vectorized (NumPy array) indexing scatters values through the buffer.
        # The View handles accumulator (n+1 dim raw array) assignment itself.
        if any(isinstance(a, np.ndarray) for a in indexes):
            self.view(flow=True)[self._compute_vectorized_index(indexes)] = np.asarray(
                value
            )
            return

        # A Histogram value must keep its flow bins; np.asarray() would call
        # __array__, which drops them (returns view(flow=False)).
        in_array = (
            value.view(flow=True) if isinstance(value, Histogram) else np.asarray(value)
        )
        view: Any = self.view(flow=True)

        value_shape: tuple[int, ...]

        # Support raw arrays for accumulators, the final dimension is the constructor values
        if (
            in_array.ndim > 0
            and len(view.dtype) > 0
            and len(in_array.dtype) == 0  # type: ignore[arg-type]
            and len(view.dtype) == in_array.shape[-1]
        ):
            value_shape = in_array.shape[:-1]
            value_ndim = in_array.ndim - 1
        else:
            value_shape = in_array.shape
            value_ndim = in_array.ndim
        value_n_slice = sum(isinstance(i, slice) for i in indexes)
        if isinstance(self._hist, _core.hist.any_multi_cell):
            # MultiCell histograms have to provide the cell index as first dimension, but the cell index is not included in the histogram indexing.
            # Slicing over the cell index is not possible for __setitem__ and is always represented as a full slice (as slice(None, None, None)).
            # Therefore, the number of slices is always one large than the indexed number of slices in the MultiCell case.
            value_n_slice += 1

        # NumPy does not broadcast partial slices, but we would need
        # to allow it (because we do allow broadcasting up dimensions)
        # Instead, we simply require matching dimensions.
        if value_ndim > 0 and value_ndim != value_n_slice:
            if isinstance(self._hist, _core.hist.any_multi_cell):
                msg = f"Setting a {len(indexes)}D MultiCell histogram with a {value_ndim}D array must have a one higher dimension of array than histogram"
            else:
                msg = f"Setting a {len(indexes)}D histogram with a {value_ndim}D array must have a matching number of dimensions"
            raise ValueError(msg)

        # Here, value_n does not increment with n if this is not a slice
        value_n = 0
        if isinstance(self._hist, _core.hist.any_multi_cell):
            # Ignore first dimension for MultiCell arrays, the first dimension is for the cells, the normal histogram axis indexing starts with dimension 2 in this case
            value_n = 1
        # value_hist_axis tracks the axis index in the value histogram (always 0-based,
        # independent of the MultiCell offset in value_n)
        value_hist_axis = 0
        for n, request in enumerate(indexes):
            has_underflow = self.axes[n].traits.underflow
            has_overflow = self.axes[n].traits.overflow

            if isinstance(request, slice):
                # This ensures that callable start/stop are handled
                start, stop = self.axes[n]._process_loc(request.start, request.stop)

                # Only consider underflow/overflow if the endpoints are not given
                use_underflow = has_underflow and start < 0
                use_overflow = has_overflow and stop > len(self.axes[n])

                # If the input is a histogram, we need to exactly match underflow/overflow
                if isinstance(value, Histogram):
                    in_underflow = value.axes[value_hist_axis].traits.underflow
                    in_overflow = value.axes[value_hist_axis].traits.overflow

                    if use_underflow != in_underflow or use_overflow != in_overflow:
                        msg = (
                            f"Cannot set histogram with underflow={in_underflow} and overflow={in_overflow} "
                            f"to a histogram slice with underflow={use_underflow} and overflow={use_overflow}"
                        )
                        raise ValueError(msg)

                # Convert to non-flow coordinates
                start_real = start + 1 if has_underflow else start
                stop_real = stop + 1 if has_underflow else stop

                # This is the total requested length without flow bins
                request_len = min(stop, len(self.axes[n])) - max(start, 0)

                # If set to a scalar, then treat it like broadcasting without flow bins
                # Normal requests here too
                # Also single element broadcasting
                if (
                    value_ndim == 0
                    or request_len == value_shape[value_n]
                    or value_shape[value_n] == 1
                ):
                    start_real += 1 if start < 0 else 0
                    stop_real -= 1 if stop > len(self.axes[n]) else 0

                # Expanded setting
                elif request_len + use_underflow + use_overflow == value_shape[value_n]:
                    pass

                else:
                    msg = f"Mismatched shapes {value_shape} in dimension {n}"
                    msg += f", {value_shape[value_n]} != {request_len}"
                    if use_underflow or use_overflow:
                        msg += f" or {request_len + use_underflow + use_overflow}"
                    raise ValueError(msg)
                logger.debug(
                    "__setitem__: axis %i, start: %i (actual %i), stop: %i (actual %i)",
                    n,
                    start,
                    start_real,
                    stop,
                    stop_real,
                )
                indexes[n] = slice(start_real, stop_real, request.step)
                value_n += 1
                value_hist_axis += 1
            else:
                indexes[n] = self._flow_pick_index(n, int(request))  # type: ignore[arg-type]

        if isinstance(self._hist, _core.hist.any_multi_cell):
            # View of multi cell histograms has as first (index 0) dimension the cell index
            # Add a full slice to the beginning of the slicing expression to adept for this cell index
            # e.g. a slice like [0, :, 3] is converted to [:, 0, :, 3]
            indexes.insert(0, slice(None, None, None))
        view[tuple(indexes)] = in_array

    def project(self, *args: int, flow: bool = True) -> Self:
        """
        Project to a single axis or several axes on a multidimensional histogram.
        Provided a list of axis numbers, this will produce the histogram over
        those axes only. Flow bins are used if available. If flow is False,
        flow bins on the integrated-out axes are excluded.
        """
        for arg in args:
            if arg < 0 or arg >= self.ndim:
                raise ValueError(
                    f"Projection axis must be a valid axis number 0 to {self.ndim - 1}, not {arg}"
                )

        if flow:
            return self._new_hist(self._hist.project(*args))

        keep_axes = set(args)
        drop_axes = [i for i in range(self.ndim) if i not in keep_axes]

        slices = [
            _core.algorithm.slice(
                i, 0, self.axes[i].size, _core.algorithm.slice_mode.crop
            )
            for i in drop_axes
        ]

        reduced_hist = self._hist.reduce(*slices) if slices else self._hist
        return self._new_hist(reduced_hist.project(*args))

    # Implementation of PlottableHistogram

    @property
    def kind(self) -> Kind:
        """
        Returns Kind.COUNT if this is a normal summing histogram, and Kind.MEAN if this is a
        mean histogram.

        :return: Kind
        """
        mean = self._hist._storage_type in {
            _core.storage.mean,
            _core.storage.weighted_mean,
        }

        return Kind.MEAN if mean else Kind.COUNT

    @typing.overload
    def values(
        self: Histogram[bhs.Int64] | Histogram[bhs.AtomicInt64], flow: bool = ...
    ) -> np.typing.NDArray[np.int64]: ...

    @typing.overload
    def values(
        self: Histogram[bhs.Double]
        | Histogram[bhs.Unlimited]
        | Histogram[bhs.Weight]
        | Histogram[bhs.Mean]
        | Histogram[bhs.WeightedMean]
        | Histogram[bhs.MultiCell],
        flow: bool = ...,
    ) -> np.typing.NDArray[np.float64]: ...

    @typing.overload
    def values(
        self, flow: bool = ...
    ) -> np.typing.NDArray[np.float64] | np.typing.NDArray[np.int64]: ...

    def values(
        self, flow: bool = False
    ) -> np.typing.NDArray[np.float64] | np.typing.NDArray[np.int64]:
        """
        Returns the accumulated values. The counts for simple histograms, the
        sum of weights for weighted histograms, the mean for profiles, etc.

        If counts is equal to 0, the value in that cell is undefined if
        kind == "MEAN".

        :param flow: Enable flow bins. Not part of PlottableHistogram, but
        included for consistency with other methods and flexibility.

        :return: "np.typing.NDArray[Any]"[np.float64]
        """

        view: Any = self.view(flow)
        # TODO: Might be a NumPy typing bug
        if len(view.dtype) == 0:
            return view  # type: ignore[no-any-return]
        return view.value  # type: ignore[no-any-return]

    @typing.overload
    def variances(
        self: Histogram[bhs.AtomicInt64] | Histogram[bhs.Int64], flow: bool = ...
    ) -> np.typing.NDArray[np.int64] | None: ...

    @typing.overload
    def variances(
        self: Histogram[bhs.Double]
        | Histogram[bhs.Unlimited]
        | Histogram[bhs.MultiCell],
        flow: bool = ...,
    ) -> np.typing.NDArray[np.float64] | None: ...

    @typing.overload
    def variances(
        self: Histogram[bhs.Weight] | Histogram[bhs.Mean] | Histogram[bhs.WeightedMean],
        flow: bool = ...,
    ) -> np.typing.NDArray[np.float64]: ...

    @typing.overload
    def variances(
        self, flow: bool = False
    ) -> np.typing.NDArray[np.int64] | np.typing.NDArray[np.float64] | None: ...

    def variances(
        self, flow: bool = False
    ) -> np.typing.NDArray[np.int64] | np.typing.NDArray[np.float64] | None:
        """
        Returns the estimated variance of the accumulated values. The sum of squared
        weights for weighted histograms, the variance of samples for profiles, etc.
        For an unweighed histogram where kind == "COUNT", this should return the same
        as values if the histogram was not filled with weights, and None otherwise.
        If counts is equal to 1 or less, the variance in that cell is undefined if
        kind == "MEAN". This must be written <= 1, and not < 2; when this
        effective counts (weighed mean), then counts could be less than 2 but
        more than 1.

        If kind == "MEAN", the counts can be used to compute the error on the mean
        as sqrt(variances / counts), this works whether or not the entries are
        weighted if the weight variance was tracked by the implementation.

        Currently, this always returns - but in the future, it will return None
        if a weighted fill is made on a unweighed storage.

        :param flow: Enable flow bins. Not part of PlottableHistogram, but
        included for consistency with other methods and flexibility.

        :return: "np.typing.NDArray[Any]"[np.float64]
        """

        view: Any = self.view(flow)
        if len(view.dtype) == 0:
            return view if self._variance_known else None

        if hasattr(view, "sum_of_weights"):
            valid = view.sum_of_weights**2 > view.sum_of_weights_squared
            return np.divide(  # type: ignore[no-any-return]
                view.variance,
                view.sum_of_weights,
                out=np.full(view.sum_of_weights.shape, np.nan),
                where=valid,
            )

        if hasattr(view, "count"):
            return np.divide(  # type: ignore[no-any-return]
                view.variance,
                view.count,
                out=np.full(view.count.shape, np.nan),
                where=view.count > 1,
            )

        return view.variance  # type: ignore[no-any-return]

    @typing.overload
    def counts(
        self: Histogram[bhs.Int64] | Histogram[bhs.AtomicInt64], flow: bool = ...
    ) -> np.typing.NDArray[np.int64]: ...

    @typing.overload
    def counts(
        self: Histogram[bhs.Double]
        | Histogram[bhs.Unlimited]
        | Histogram[bhs.Weight]
        | Histogram[bhs.Mean]
        | Histogram[bhs.WeightedMean]
        | Histogram[bhs.MultiCell],
        flow: bool = ...,
    ) -> np.typing.NDArray[np.float64]: ...

    @typing.overload
    def counts(
        self, flow: bool = ...
    ) -> np.typing.NDArray[np.float64] | np.typing.NDArray[np.int64]: ...

    def counts(
        self, flow: bool = False
    ) -> np.typing.NDArray[np.float64] | np.typing.NDArray[np.int64]:
        """
        Returns the number of entries in each bin for an unweighted
        histogram or profile and an effective number of entries (defined below)
        for a weighted histogram or profile. An exotic generalized histogram could
        have no sensible .counts, so this is Optional and should be checked by
        Consumers.

        If kind == "MEAN", counts (effective or not) can and should be used to
        determine whether the mean value and its variance should be displayed
        (see documentation of values and variances, respectively). The counts
        should also be used to compute the error on the mean (see documentation
        of variances).

        For a weighted histogram, counts is defined as sum_of_weights ** 2 /
        sum_of_weights_squared. It is equal or less than the number of times
        the bin was filled, the equality holds when all filled weights are equal.
        The larger the spread in weights, the smaller it is, but it is always 0
        if filled 0 times, and 1 if filled once, and more than 1 otherwise.

        :return: "np.typing.NDArray[Any]"[np.float64]
        """

        view: Any = self.view(flow)

        if len(view.dtype) == 0:
            return view  # type: ignore[no-any-return]

        if hasattr(view, "sum_of_weights"):
            return np.divide(  # type: ignore[no-any-return]
                view.sum_of_weights**2,
                view.sum_of_weights_squared,
                out=np.zeros_like(view.sum_of_weights, dtype=np.float64),
                where=view.sum_of_weights_squared != 0,
            )

        if hasattr(view, "count"):
            return view.count  # type: ignore[no-any-return]

        return view.value  # type: ignore[no-any-return]


if TYPE_CHECKING:
    from uhi.typing.plottable import PlottableHistogram

    _: PlottableHistogram = typing.cast(Histogram[Any], None)
