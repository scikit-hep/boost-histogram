from __future__ import annotations

import collections.abc
import copy
import logging
import sys
import threading
import typing
import warnings
from collections.abc import Iterable, Mapping
from enum import Enum
from os import cpu_count
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    NewType,
    SupportsIndex,
    TypeVar,
    Union,
)

import numpy as np

import boost_histogram
from boost_histogram import _core

from . import serialization
from ._compat.typing import Self
from ._utils import cast, register
from .axis import AxesTuple, Axis, Variable
from .storage import Double, Storage
from .typing import Accumulator, ArrayLike, CppHistogram, RebinProtocol
from .view import MeanView, WeightedMeanView, WeightedSumView, _to_view

if TYPE_CHECKING:
    from builtins import ellipsis


try:
    from . import _core
except ImportError as err:
    if "_core" not in str(err):
        raise

    new_msg = "Did you forget to compile boost-histogram? Use CMake or scikit-build-core to build, see the readme."

    if sys.version_info >= (3, 11):
        err.add_note(new_msg)
        raise

    total_msg = f"{err}\n{new_msg}"
    new_exception = type(err)(new_msg, name=err.name, path=err.path)
    raise new_exception from err


# This is a StrEnum as defined in Python 3.10
class Kind(str, Enum):
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


# Support cloudpickle - pybind11 submodules do not have __file__ attributes
# And setting this in C++ causes a segfault
_core.accumulators.__file__ = _core.__file__
_core.algorithm.__file__ = _core.__file__
_core.axis.__file__ = _core.__file__
_core.axis.transform.__file__ = _core.__file__
_core.hist.__file__ = _core.__file__
_core.storage.__file__ = _core.__file__


NOTHING = object()


_histograms: set[type[CppHistogram]] = {
    _core.hist.any_double,
    _core.hist.any_int64,
    _core.hist.any_atomic_int64,
    _core.hist.any_unlimited,
    _core.hist.any_weight,
    _core.hist.any_mean,
    _core.hist.any_weighted_mean,
}

logger = logging.getLogger(__name__)


CppAxis = NewType("CppAxis", object)

SimpleIndexing = Union[SupportsIndex, slice, RebinProtocol]
InnerIndexing = Union[SimpleIndexing, Callable[[Axis], int]]
FullInnerIndexing = Union[InnerIndexing, list[InnerIndexing]]
IndexingWithMapping = Union[FullInnerIndexing, Mapping[int, FullInnerIndexing]]
IndexingExpr = Union[IndexingWithMapping, tuple[IndexingWithMapping, ...], "ellipsis"]

T = TypeVar("T")


def _fill_cast(
    value: T, *, inner: bool = False
) -> T | np.typing.NDArray[Any] | tuple[T, ...]:
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
    seqs = (collections.abc.Sequence, np.ndarray)
    msg1 = f"Sample key-argument needs to be a sequence, {sample.__class__.__name__} given."
    if isinstance(sample, str) and not isinstance(sample, seqs):
        raise ValueError(msg1)
    sample_dim = np.array(sample).ndim
    msg2 = f"Sample key-argument needs to be 1 dimensional, {sample_dim} given."
    if sample_dim != 1:
        raise ValueError(msg2)


def _arg_shortcut(item: tuple[int, float, float] | Axis | CppAxis) -> CppAxis:
    if isinstance(item, tuple) and len(item) == 3:
        msg = "Using () directly in constructor is a developer shortcut and will be removed in a future version"
        warnings.warn(msg, FutureWarning, stacklevel=4)
        return _core.axis.regular_uoflow(item[0], item[1], item[2])  # type: ignore[return-value]

    if isinstance(item, Axis):
        return item._ax  # type: ignore[no-any-return]

    raise TypeError("Only axes supported in histogram constructor")


def _expand_ellipsis(indexes: Iterable[Any], rank: int) -> list[Any]:
    indexes = list(indexes)
    number_ellipses = indexes.count(Ellipsis)
    if number_ellipses == 0:
        return indexes
    if number_ellipses == 1:
        index = indexes.index(Ellipsis)
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
    Combine two views into one, in-place. This is used for threaded filling.
    """
    pos = [slice(None)] * (i)
    if new_view.dtype.names:
        for field in new_view.dtype.names:
            new_view[(*pos, jj, ...)][field] += reduced_view[(*pos, j, ...)][field]
    else:
        new_view[(*pos, jj, ...)] += reduced_view[(*pos, j, ...)]


H = TypeVar("H", bound="Histogram")


# We currently do not cast *to* a histogram, but this is consistent
# and could be used later.
@register(_histograms)  # type: ignore[arg-type]
class Histogram:
    # Note this is a __slots__ __dict__ class!
    __slots__ = (
        "__dict__",
        "_hist",
        "axes",
    )
    # .metadata and ._variance_known are part of the dict

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
    def __init__(self, arg: Histogram, /, *, metadata: Any = ...) -> None: ...

    @typing.overload
    def __init__(self, arg: dict[str, Any], /, *, metadata: Any = ...) -> None: ...

    @typing.overload
    def __init__(self, arg: CppHistogram, /, *, metadata: Any = ...) -> None: ...

    @typing.overload
    def __init__(
        self,
        *axes: Axis | CppAxis,
        storage: Storage = ...,
        metadata: Any = ...,
    ) -> None: ...

    def __init__(
        self,
        *axes: Axis | CppAxis | Histogram | CppHistogram | dict[str, Any],
        storage: Storage | None = None,
        metadata: Any = None,
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
            Data that is passed along if a new histogram is created
        """
        self._variance_known = True
        storage_err_msg = "storage= is not allowed with conversion constructor"

        # Allow construction from a raw histogram object (internal)
        if len(axes) == 1 and isinstance(axes[0], tuple(_histograms)):
            if storage is not None:
                raise TypeError(storage_err_msg)
            cpp_hist: CppHistogram = axes[0]  # type: ignore[assignment]
            self._from_histogram_cpp(cpp_hist, metadata=None)
            return

        # If we construct with another Histogram as the only positional argument,
        # support that too
        if len(axes) == 1 and isinstance(axes[0], Histogram):
            if storage is not None:
                raise TypeError(storage_err_msg)
            self._from_histogram_object(axes[0], metadata=metadata)
            return

        # Support objects that provide a to_boost method, like Uproot
        if len(axes) == 1 and hasattr(axes[0], "_to_boost_histogram_"):
            if storage is not None:
                raise TypeError(storage_err_msg)
            self._from_histogram_object(
                axes[0]._to_boost_histogram_(), metadata=metadata
            )
            return

        # Support UHI
        if len(axes) == 1 and isinstance(axes[0], dict) and "uhi_schema" in axes[0]:
            if storage is not None:
                raise TypeError(storage_err_msg)
            self._from_histogram_object(
                serialization.from_uhi(axes[0]), metadata=metadata
            )
            return

        if storage is None:
            storage = Double()

        self.metadata = metadata

        # Check for missed parenthesis or incorrect types
        if not isinstance(storage, Storage):
            msg_storage = (  # type: ignore[unreachable]
                "Passing in an initialized storage has been removed. Please add ()."
            )
            msg_unknown = "Only storages allowed in storage argument"
            raise KeyError(msg_storage if issubclass(storage, Storage) else msg_unknown)

        # Allow a tuple to represent a regular axis
        axes = tuple(_arg_shortcut(arg) for arg in axes)  # type: ignore[arg-type]

        if len(axes) > _core.hist._axes_limit:
            msg = f"Too many axes, must be less than {_core.hist._axes_limit}"
            raise IndexError(msg)

        # Check all available histograms, and if the storage matches, return that one
        for h in _histograms:
            if isinstance(storage, h._storage_type):
                self._hist = h(axes, storage)  # type: ignore[arg-type]
                self.axes = self._generate_axes_()
                return

        raise TypeError("Unsupported storage")

    @classmethod
    def _clone(
        cls,
        _hist: Histogram | CppHistogram,
        *,
        other: Histogram | None = None,
        memo: Any = NOTHING,
    ) -> Self:
        """
        Clone a histogram (possibly of a different base). Does not trigger __init__.
        This will copy data from `other=` if non-None, otherwise metadata gets copied from the input.
        """

        self = cls.__new__(cls)
        if isinstance(_hist, tuple(_histograms)):
            self._from_histogram_cpp(_hist)  # type: ignore[arg-type]
            if other is not None:
                return cls._clone(self, other=other, memo=memo)
            return self

        assert isinstance(_hist, Histogram)

        if other is None:
            other = _hist

        self._from_histogram_object(_hist)

        if memo is NOTHING:
            self.__dict__ = copy.copy(other.__dict__)
        else:
            self.__dict__ = copy.deepcopy(other.__dict__, memo)

        for ax in self.axes:
            if memo is NOTHING:
                ax.__dict__ = copy.copy(ax._ax.raw_metadata)
            else:
                ax.__dict__ = copy.deepcopy(ax._ax.raw_metadata, memo)
        return self

    def _new_hist(self, _hist: CppHistogram, memo: Any = NOTHING) -> Self:
        """
        Return a new histogram given a new _hist, copying current metadata.
        """
        return self.__class__._clone(_hist, other=self, memo=memo)

    def _from_histogram_cpp(self, other: CppHistogram, *, metadata: Any = None) -> None:
        """
        Import a Cpp histogram.
        """
        self._variance_known = True
        self._hist = other
        self.metadata = metadata
        self.axes = self._generate_axes_()

    def _from_histogram_object(self, other: Histogram, *, metadata: Any = None) -> None:
        """
        Convert self into a new histogram object based on another, possibly
        converting from a different subclass.
        """
        self._hist = other._hist
        self.__dict__ = copy.copy(other.__dict__)
        self.axes = self._generate_axes_()
        for ax in self.axes:
            ax.__dict__ = copy.copy(ax._ax.raw_metadata)
        self.metadata = other.metadata if metadata is None else metadata

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
    def _export_bh_(cls, self: Histogram) -> None:
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

    def view(
        self, flow: bool = False
    ) -> np.typing.NDArray[Any] | WeightedSumView | WeightedMeanView | MeanView:
        """
        Return a view into the data, optionally with overflow turned on.
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
        return np.asarray(self.view(False), dtype=dtype, **kwargs)  # type: ignore[call-overload]

    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        return hasattr(other, "_hist") and self._hist == other._hist

    def __ne__(self, other: object) -> bool:
        return (not hasattr(other, "_hist")) or self._hist != other._hist

    def __add__(self, other: Histogram | np.typing.NDArray[Any] | float) -> Self:
        result = self.copy(deep=False)
        return result.__iadd__(other)

    def __iadd__(self, other: Histogram | np.typing.NDArray[Any] | float) -> Self:
        if isinstance(other, (int, float)) and other == 0:
            return self
        self._compute_inplace_op("__iadd__", other)

        # Addition may change the axes if they can grow
        self.axes = self._generate_axes_()

        return self

    def __radd__(self, other: np.typing.NDArray[Any] | float) -> Self:
        return self + other

    def __sub__(self, other: Histogram | np.typing.NDArray[Any] | float) -> Self:
        result = self.copy(deep=False)
        return result.__isub__(other)

    def __isub__(self, other: Histogram | np.typing.NDArray[Any] | float) -> Self:
        if isinstance(other, (int, float)) and other == 0:
            return self
        self._compute_inplace_op("__isub__", other)

        self.axes = self._generate_axes_()

        return self

    # If these fail, the underlying object throws the correct error
    def __mul__(self, other: Histogram | np.typing.NDArray[Any] | float) -> Self:
        result = self.copy(deep=False)
        return result._compute_inplace_op("__imul__", other)

    def __rmul__(self, other: np.typing.NDArray[Any] | float) -> Self:
        return self * other

    def __truediv__(self, other: Histogram | np.typing.NDArray[Any] | float) -> Self:
        result = self.copy(deep=False)
        return result._compute_inplace_op("__itruediv__", other)

    def __div__(self, other: Histogram | np.typing.NDArray[Any] | float) -> Self:
        result = self.copy(deep=False)
        return result._compute_inplace_op("__idiv__", other)

    def __idiv__(self, other: Histogram | np.typing.NDArray[Any] | float) -> Self:
        return self._compute_inplace_op("__idiv__", other)

    def __itruediv__(self, other: Histogram | np.typing.NDArray[Any] | float) -> Self:
        return self._compute_inplace_op("__itruediv__", other)

    def __imul__(self, other: Histogram | np.typing.NDArray[Any] | float) -> Self:
        return self._compute_inplace_op("__imul__", other)

    def _compute_inplace_op(
        self, name: str, other: Histogram | np.typing.NDArray[Any] | float
    ) -> Self:
        # Also takes CppHistogram, but that confuses mypy because it's hard to pick out
        if isinstance(other, Histogram):
            getattr(self._hist, name)(other._hist)
        elif isinstance(other, tuple(_histograms)):
            getattr(self._hist, name)(other)
        elif hasattr(other, "shape") and other.shape:
            assert not isinstance(other, float)

            if len(other.shape) != self.ndim:
                msg = f"Number of dimensions {len(other.shape)} must match histogram {self.ndim}"
                raise ValueError(msg)

            if all(a in {b, 1} for a, b in zip(other.shape, self.shape)):
                view = self.view(flow=False)
                getattr(view, name)(other)
            elif all(a in {b, 1} for a, b in zip(other.shape, self.axes.extent)):
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
            available threads (usually two per core).
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

        # Convert to NumPy arrays
        args_ars = _fill_cast(args)
        weight_ars = _fill_cast(weight)
        sample_ars = _fill_cast(sample)

        if threads == 0:
            threads = cpu_count()

        if threads is None or threads == 1:
            self._hist.fill(*args_ars, weight=weight_ars, sample=sample_ars)
            return self

        if self._hist._storage_type in {
            _core.storage.mean,
            _core.storage.weighted_mean,
        }:
            raise RuntimeError("Mean histograms do not support threaded filling")

        data: list[list[np.typing.NDArray[Any]] | list[str]] = [
            np.array_split(a, threads) if not isinstance(a, str) else [a] * threads
            for a in args_ars
        ]

        weights: list[Any]
        if weight is None or np.isscalar(weight):
            assert threads is not None
            weights = [weight_ars] * threads
        else:
            weights = np.array_split(weight_ars, threads)

        samples: list[Any]
        if sample_ars is None or np.isscalar(sample_ars):
            assert threads is not None
            samples = [sample_ars] * threads
        else:
            samples = np.array_split(sample_ars, threads)

        if self._hist._storage_type is _core.storage.atomic_int64:

            def fun(
                weight: ArrayLike | None,
                sample: ArrayLike | None,
                *args: np.typing.NDArray[Any],
            ) -> None:
                self._hist.fill(*args, weight=weight, sample=sample)

        else:
            sum_lock = threading.Lock()

            def fun(
                weight: ArrayLike | None,
                sample: ArrayLike | None,
                *args: np.typing.NDArray[Any],
            ) -> None:
                local_hist = copy.copy(self._hist)
                local_hist.reset()
                local_hist.fill(*args, weight=weight, sample=sample)
                with sum_lock:
                    self._hist += local_hist

        thread_list = [
            threading.Thread(target=fun, args=arrays)
            for arrays in zip(weights, samples, *data)
        ]

        for thread in thread_list:
            thread.start()

        for thread in thread_list:
            thread.join()

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
    def storage_type(self) -> type[Storage]:
        return cast(self, self._hist._storage_type, Storage)  # type: ignore[return-value]

    @property
    def _storage_type(self) -> type[Storage]:
        warnings.warn(
            "Accessing storage type has changed from _storage_type to storage_type, and will be removed in future.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cast(self, self._hist._storage_type, Storage)  # type: ignore[return-value]

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
                self._hist.axis(i).raw_metadata = {
                    "metadata": self._hist.axis(i).raw_metadata
                }

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
        ret += f"{sep}{storage_newline}storage={self.storage_type()}"  # pylint: disable=not-callable
        ret += ")"
        outer = self.sum(flow=True)
        if outer:
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
        if index is sum or hasattr(index, "factor"):  # type: ignore[comparison-overlap]
            return slice(None, None, index)

        # General locators
        # Note that MyPy doesn't like these very much - the fix
        # will be to properly set input types
        if callable(index):
            return index(self.axes[axis])

        if isinstance(index, float):
            raise TypeError(f"Index {index} must be an integer, not float")

        if isinstance(index, SupportsIndex):
            if abs(int(index)) >= self._hist.axis(axis).size:
                raise IndexError("histogram index is out of range")
            return int(index) % self._hist.axis(axis).size

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

    def to_numpy(
        self, flow: bool = False, *, dd: bool = False, view: bool = False
    ) -> (
        tuple[np.typing.NDArray[Any], ...]
        | tuple[np.typing.NDArray[Any], tuple[np.typing.NDArray[Any], ...]]
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

        hist, *edges = self._hist.to_numpy(flow)
        hist = self.view(flow=flow) if view else self.values(flow=flow)

        return (hist, edges) if dd else (hist, *edges)

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

    def sum(self, flow: bool = False) -> float | Accumulator:
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

    # TODO: Marked as too complex by flake8. Should be factored out a bit.
    def __getitem__(self, index: IndexingExpr) -> Self | float | Accumulator:
        indexes = self._compute_commonindex(index)

        # If this is (now) all integers, return the bin contents
        # But don't try *dict!
        if not hasattr(indexes, "items") and all(
            isinstance(a, SupportsIndex) for a in indexes
        ):
            return self._hist.at(*indexes)  # type: ignore[no-any-return, arg-type]

        integrations: set[int] = set()
        slices: list[_core.algorithm.reduce_command] = []
        pick_each: dict[int, int] = {}
        pick_set: dict[int, list[int]] = {}
        reduced: CppHistogram | None = None

        # Compute needed slices and projections
        for i, ind in enumerate(indexes):  # pylint: disable=too-many-nested-blocks
            if isinstance(ind, SupportsIndex):
                pick_each[i] = ind.__index__() + (
                    1 if self.axes[i].traits.underflow else 0
                )
                continue

            if isinstance(ind, collections.abc.Sequence):
                pick_set[i] = list(ind)
                continue

            if not isinstance(ind, slice):
                raise IndexError(
                    "Must be a slice, an integer, or follow the locator protocol."
                )

            # If the dictionary brackets are forgotten, it's easy to put a slice
            # into a slice - adding a nicer error message in that case
            if any(isinstance(v, slice) for v in (ind.start, ind.stop, ind.step)):
                raise TypeError(
                    "You have put a slice in a slice. Did you forget curly braces [{...}]?"
                )

            # This ensures that callable start/stop are handled
            start, stop = self.axes[i]._process_loc(ind.start, ind.stop)

            groups = []
            new_axis = None
            if ind != slice(None):
                merge = 1
                if ind.step is not None:
                    if getattr(ind.step, "factor", None) is not None:
                        merge = ind.step.factor
                    elif (
                        hasattr(ind.step, "axis_mapping")
                        and (tmp_both := ind.step.axis_mapping(self.axes[i]))
                        is not None
                    ):
                        groups, new_axis = tmp_both
                    elif (
                        hasattr(ind.step, "group_mapping")
                        and (tmp_groups := ind.step.group_mapping(self.axes[i]))
                        is not None
                    ):
                        groups = tmp_groups
                    elif callable(ind.step):
                        if ind.step is sum:
                            integrations.add(i)
                        else:
                            raise NotImplementedError

                        if ind.start is not None or ind.stop is not None:
                            slices.append(
                                _core.algorithm.slice(
                                    i, start, stop, _core.algorithm.slice_mode.crop
                                )
                            )
                        if len(groups) == 0:
                            continue
                    else:
                        raise IndexError(
                            "The third argument to a slice must be rebin or projection"
                        )

                assert isinstance(start, int)
                assert isinstance(stop, int)
                # rebinning with factor
                if len(groups) == 0:
                    slices.append(
                        _core.algorithm.slice_and_rebin(i, start, stop, merge)
                    )
                # rebinning with groups
                elif len(groups) != 0:
                    if not reduced:
                        reduced = self._hist
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

                    new_reduced = reduced.__class__(axes)
                    new_view = new_reduced.view(flow=True)
                    j = 0
                    new_j_base = 0

                    if old_axis.traits_underflow and axes[i].traits_underflow:
                        groups.insert(0, 1)
                    elif axes[i].traits_underflow:
                        new_j_base = 1

                    if old_axis.traits_overflow and axes[i].traits_overflow:
                        groups.append(1)

                    for new_j, group in enumerate(groups):
                        for _ in range(group):
                            _combine_group_contents(
                                new_view, reduced_view, i, j, new_j + new_j_base
                            )
                            j += 1

                        if (
                            old_axis.traits_underflow
                            and not axes[i].traits_ordered
                            and axes[i].traits_overflow
                        ):
                            _combine_group_contents(new_view, reduced_view, i, 0, -1)

                    reduced = new_reduced

        # Will be updated below
        if (slices or pick_set or pick_each or integrations) and not reduced:
            reduced = self._hist
        elif not reduced:
            reduced = copy.copy(self._hist)

        if pick_each:
            tuple_slice = tuple(
                pick_each.get(i, slice(None)) for i in range(reduced.rank())
            )
            logger.debug("Slices for pick each: %s", tuple_slice)
            axes = [
                reduced.axis(i) for i in range(reduced.rank()) if i not in pick_each
            ]
            logger.debug("Axes: %s", axes)
            new_reduced = reduced.__class__(axes)
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
            for i in pick_set:  # pylint: disable=consider-using-dict-items
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

        in_array = np.asarray(value)
        view: Any = self.view(flow=True)

        value_shape: tuple[int, ...]

        # Support raw arrays for accumulators, the final dimension is the constructor values
        if (
            in_array.ndim > 0
            and len(view.dtype) > 0
            and len(in_array.dtype) == 0
            and len(view.dtype) == in_array.shape[-1]
        ):
            value_shape = in_array.shape[:-1]
            value_ndim = in_array.ndim - 1
        else:
            value_shape = in_array.shape
            value_ndim = in_array.ndim

        # NumPy does not broadcast partial slices, but we would need
        # to allow it (because we do allow broadcasting up dimensions)
        # Instead, we simply require matching dimensions.
        if value_ndim > 0 and value_ndim != sum(isinstance(i, slice) for i in indexes):
            msg = f"Setting a {len(indexes)}D histogram with a {value_ndim}D array must have a matching number of dimensions"
            raise ValueError(msg)

        # Here, value_n does not increment with n if this is not a slice
        value_n = 0
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
                    in_underflow = value.axes[n].traits.underflow
                    in_overflow = value.axes[n].traits.overflow

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
                    msg = f"Mismatched shapes in dimension {n}"
                    msg += f", {value_shape[n]} != {request_len}"
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
            else:
                indexes[n] = request + has_underflow

        view[tuple(indexes)] = in_array

    def project(self, *args: int) -> Self | float | Accumulator:
        """
        Project to a single axis or several axes on a multidimensional histogram.
        Provided a list of axis numbers, this will produce the histogram over
        those axes only. Flow bins are used if available.
        """
        for arg in args:
            if arg < 0 or arg >= self.ndim:
                raise ValueError(
                    f"Projection axis must be a valid axis number 0 to {self.ndim - 1}, not {arg}"
                )

        return self._new_hist(self._hist.project(*args))

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

    def values(self, flow: bool = False) -> np.typing.NDArray[Any]:
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
            return view
        return view.value

    def variances(self, flow: bool = False) -> np.typing.NDArray[Any] | None:
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
            return np.divide(
                view.variance,
                view.sum_of_weights,
                out=np.full(view.sum_of_weights.shape, np.nan),
                where=valid,
            )

        if hasattr(view, "count"):
            return np.divide(
                view.variance,
                view.count,
                out=np.full(view.count.shape, np.nan),
                where=view.count > 1,
            )

        return view.variance

    def counts(self, flow: bool = False) -> np.typing.NDArray[Any]:
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
            return view

        if hasattr(view, "sum_of_weights"):
            return np.divide(
                view.sum_of_weights**2,
                view.sum_of_weights_squared,
                out=np.zeros_like(view.sum_of_weights, dtype=np.float64),
                where=view.sum_of_weights_squared != 0,
            )

        if hasattr(view, "count"):
            return view.count

        return view.value


if TYPE_CHECKING:
    from uhi.typing.plottable import PlottableHistogram

    _: PlottableHistogram = typing.cast(Histogram, None)
