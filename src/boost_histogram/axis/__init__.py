from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Callable,
    ClassVar,
    Literal,
    TypedDict,
    TypeVar,
    Union,
)

import numpy as np  # pylint: disable=unused-import

import boost_histogram

from .._compat.typing import Self
from .._core import axis as ca
from .._utils import cast, register, zip_strict
from . import transform
from .transform import AxisTransform

__all__ = [
    "ArrayTuple",
    "AxesTuple",
    "Axis",
    "Boolean",
    "IntCategory",
    "Integer",
    "Regular",
    "StrCategory",
    "Traits",
    "Variable",
    "transform",
]


def __dir__() -> list[str]:
    return __all__


def _isstr(value: Any) -> bool:
    """
    Check to see if this is a stringlike or a (nested) iterable of stringlikes
    """

    if isinstance(value, (str, bytes)):
        return True
    if hasattr(value, "__iter__"):
        return all(_isstr(v) for v in value)
    return False


def _opts(**kwargs: bool) -> set[str]:
    return {k for k, v in kwargs.items() if v}


AxCallOrInt = Union[int, Callable[["Axis"], int]]


@dataclass(order=True, frozen=True)
class Traits:
    underflow: bool = False
    overflow: bool = False
    circular: bool = False
    growth: bool = False
    continuous: bool = False
    ordered: bool = False

    @property
    def discrete(self) -> bool:
        "True if axis is not continuous"
        return not self.continuous


T = TypeVar("T", bound="Axis")


# Contains common methods and properties to all axes
class Axis:
    __slots__ = ("__dict__", "_ax")
    _family: object

    def __init_subclass__(cls, *, family: object) -> None:
        super().__init_subclass__()
        cls._family = family

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr == "__dict__":
            self._ax.raw_metadata = value
        object.__setattr__(self, attr, value)

    def __getattr__(self, attr: str) -> Any:
        if attr == "metadata":
            return
        raise AttributeError(
            f"object {self.__class__.__name__} has no attribute {attr}"
        )

    def __init__(
        self,
        ax: Any,
        metadata: dict[str, Any] | None,
        __dict__: dict[str, Any] | None,
    ) -> None:
        """
        ax: the C++ object
        metadata: the metadata keyword contents
        __dict__: the __dict__ keyword contents
        """

        self._ax = ax

        if __dict__ is not None and metadata is not None:
            raise KeyError(
                "Cannot provide metadata by keyword and __dict__, use __dict__ only"
            )
        if __dict__ is not None:
            self._ax.raw_metadata = __dict__
        elif metadata is not None:
            self._ax.raw_metadata["metadata"] = metadata

        self.__dict__ = self._ax.raw_metadata

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._ax = state["_ax"]
        self.__dict__ = self._ax.raw_metadata

    def __getstate__(self) -> dict[str, Any]:
        return {"_ax": self._ax}

    def __copy__(self) -> Self:
        other: Self = self.__class__.__new__(self.__class__)
        other._ax = copy.copy(self._ax)
        other.__dict__ = other._ax.raw_metadata
        return other

    def index(self, value: float | str) -> int:
        """
        Return the fractional index(es) given a value (or values) on the axis.
        """

        if _isstr(value):
            msg = f"index({value}) cannot be a string for a numerical axis"
            raise TypeError(msg)

        return self._ax.index(value)  # type: ignore[no-any-return]

    def value(self, index: float) -> float:
        """
        Return the value(s) given an (fractional) index (or indices).
        """

        return self._ax.value(index)  # type: ignore[no-any-return]

    def bin(self, index: float) -> int | str | tuple[float, float]:
        """
        Return the edges of the bins as a tuple for a
        continuous axis or the bin value for a
        non-continuous axis, when given an index.
        """

        return self._ax.bin(index)  # type: ignore[no-any-return]

    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        return hasattr(other, "_ax") and self._ax == other._ax

    def __ne__(self, other: object) -> bool:
        return (not hasattr(other, "_ax")) or self._ax != other._ax

    @classmethod
    def _convert_cpp(cls, cpp_object: Any) -> Self:
        nice_ax: Self = cls.__new__(cls)
        nice_ax._ax = cpp_object
        nice_ax.__dict__ = cpp_object.raw_metadata
        return nice_ax

    def __len__(self) -> int:
        return self._ax.size  # type: ignore[no-any-return]

    def __iter__(
        self,
    ) -> Iterator[float] | Iterator[str] | Iterator[tuple[float, float]]:
        return self._ax.__iter__()  # type: ignore[no-any-return]

    def _process_callable(self, value: AxCallOrInt | None, *, default: int) -> int:
        """
        This processes a callable in start or stop. None gets replaced by default.
        """
        if value is None:
            return default
        if callable(value):
            return value(self)
        return value

    def _process_loc(
        self, start: AxCallOrInt | None, stop: AxCallOrInt | None
    ) -> tuple[int, int]:
        """
        Compute start and stop into actual start and stop values in Boost.Histogram.
        None -> -1 or 0 for start, -> len or len+1 for stop. If start or stop are
        callable, then call them with the axes.

        For a non-ordered axes, flow is all or nothing, so this will ensure overflow
        is turned off if underflow is not None.
        """

        underflow = -1 if self._ax.traits_underflow else 0
        overflow = 1 if self._ax.traits_overflow else 0

        # Non-ordered axes only use flow if integrating from None to None
        if not self._ax.traits_ordered and not (start is None and stop is None):
            overflow = 0

        begin = self._process_callable(start, default=underflow)
        end = self._process_callable(stop, default=len(self) + overflow)

        return begin, end

    def __repr__(self) -> str:
        arg_str = ", ".join(self._repr_args_())
        return f"{self.__class__.__name__}({arg_str})"

    def _repr_args_(self) -> list[str]:
        """
        Return arg options for use in the repr as strings.
        """

        ret = []
        if self.metadata is not None:
            if isinstance(self.metadata, str):
                ret.append(f"metadata={self.metadata!r}")
            else:
                ret.append("metadata=...")
        return ret

    @property
    def traits(self) -> Traits:
        """
        Get traits for the axis - read only properties of a specific axis.
        """
        return Traits(
            self._ax.traits_underflow,
            self._ax.traits_overflow,
            self._ax.traits_circular,
            self._ax.traits_growth,
            self._ax.traits_continuous,
            self._ax.traits_ordered,
        )

    @property
    def size(self) -> int:
        """
        Return number of bins excluding under- and overflow.
        """
        return self._ax.size  # type: ignore[no-any-return]

    @property
    def extent(self) -> int:
        """
        Return number of bins including under- and overflow.
        """
        return self._ax.extent  # type: ignore[no-any-return]

    def __getitem__(self, i: AxCallOrInt) -> int | str | tuple[float, float]:
        """
        Access a bin, using normal Python syntax for wraparound.
        """
        # UHI support
        if callable(i):
            i = i(self)
        else:
            if i < 0:
                i += self._ax.size
            if i >= self._ax.size:
                raise IndexError(
                    f"Out of range access, {i} is more than {self._ax.size}"
                )
        assert not callable(i)
        return self.bin(i)

    @property
    def edges(self) -> np.typing.NDArray[Any]:
        return self._ax.edges

    @property
    def centers(self) -> np.typing.NDArray[Any]:
        """
        An array of bin centers.
        """
        return self._ax.centers

    @property
    def widths(self) -> np.typing.NDArray[Any]:
        """
        An array of bin widths.
        """
        return self._ax.widths


# Contains all common methods and properties for Regular axes
@register(
    {
        ca.regular_uoflow,
        ca.regular_uoflow_growth,
        ca.regular_uflow,
        ca.regular_oflow,
        ca.regular_none,
        ca.regular_pow,
        ca.regular_trans,
        ca.regular_circular,
    }
)
class Regular(Axis, family=boost_histogram):
    __slots__ = ()

    def __init__(
        self,
        bins: int,
        start: float,
        stop: float,
        *,
        metadata: Any = None,
        underflow: bool = True,
        overflow: bool = True,
        growth: bool = False,
        circular: bool = False,
        transform: AxisTransform | None = None,  # pylint: disable=redefined-outer-name
        __dict__: dict[str, Any] | None = None,
    ):
        """
        Make a regular axis with nice keyword arguments for underflow,
        overflow, and growth.

        Parameters
        ----------
        bins : int
            The number of bins between start and stop
        start : float
            The beginning value for the axis
        stop : float
            The ending value for the axis
        metadata : Any
            Fills .metadata on the axis.
        underflow : bool = True
            Enable the underflow bin
        overflow : bool = True
            Enable the overflow bin
        growth : bool = False
            Allow the axis to grow if a value is encountered out of range.
            Be careful, the axis will grow as large as needed.
        circular : bool = False
            Filling wraps around.
        transform : Optional[AxisTransform] = None
            Transform the regular bins (Log, Sqrt, and Pow(v))
        __dict__: Optional[dict[str, Any]] = None
            The full metadata dictionary
        """

        options = _opts(
            underflow=underflow, overflow=overflow, growth=growth, circular=circular
        )

        ax: ca._BaseRegular

        if transform is not None:
            if options != {"underflow", "overflow"}:
                raise KeyError("Transform supplied, cannot change other options")

            if (
                not isinstance(transform, AxisTransform)
                and AxisTransform in transform.__bases__  # type: ignore[unreachable]
            ):
                raise TypeError(f"You must pass an instance, use {transform}()")

            ax = transform._produce(bins, start, stop)

        elif options == {"growth", "underflow", "overflow"}:
            ax = ca.regular_uoflow_growth(bins, start, stop)
        elif options == {"underflow", "overflow"}:
            ax = ca.regular_uoflow(bins, start, stop)
        elif options == {"underflow"}:
            ax = ca.regular_uflow(bins, start, stop)
        elif options == {"overflow"}:
            ax = ca.regular_oflow(bins, start, stop)
        elif options in (
            {"circular", "underflow", "overflow"},
            {"circular", "overflow"},
        ):
            # growth=True, underflow=False is also correct
            ax = ca.regular_circular(bins, start, stop)

        elif options == set():
            ax = ca.regular_none(bins, start, stop)
        else:
            raise KeyError("Unsupported collection of options")

        super().__init__(ax, metadata, __dict__)

    def _repr_args_(self) -> list[str]:
        "Return inner part of signature for use in repr"

        ret = [f"{self.size:g}", f"{self.edges[0]:g}", f"{self.edges[-1]:g}"]

        if self.traits.growth:
            ret.append("growth=True")
        elif self.traits.circular:
            ret.append("circular=True")
        else:
            if not self.traits.underflow:
                ret.append("underflow=False")
            if not self.traits.overflow:
                ret.append("overflow=False")

        if self.transform is not None:
            ret.append(f"transform={self.transform}")

        ret += super()._repr_args_()

        return ret

    @property
    def transform(self) -> AxisTransform | None:
        if hasattr(self._ax, "transform"):
            return cast(self, self._ax.transform, AxisTransform)
        return None


@register(
    {
        ca.variable_none,
        ca.variable_uflow,
        ca.variable_oflow,
        ca.variable_uoflow,
        ca.variable_uoflow_growth,
        ca.variable_circular,
    }
)
class Variable(Axis, family=boost_histogram):
    __slots__ = ()

    def __init__(
        self,
        edges: Iterable[float],
        *,
        metadata: Any = None,
        underflow: bool = True,
        overflow: bool = True,
        growth: bool = False,
        circular: bool = False,
        __dict__: dict[str, Any] | None = None,
    ):
        """
        Make an axis with irregularly spaced bins. Provide a list
        or array of bin edges, and len(edges)-1 bins will be made.

        Parameters
        ----------
        edges : Array[float]
            The edges for the bins. There will be one less bin than edges.
        metadata : object
            Any Python object to attach to the axis, like a label.
        underflow : bool = True
            Enable the underflow bin
        overflow : bool = True
            Enable the overflow bin
        circular : bool = False
            Enable wraparound
        growth : bool = False
            Allow the axis to grow if a value is encountered out of range.
            Be careful, the axis will grow as large as needed.
        __dict__: Optional[dict[str, Any]] = None
            The full metadata dictionary
        """

        options = _opts(
            underflow=underflow, overflow=overflow, growth=growth, circular=circular
        )

        ax: ca._BaseVariable
        if options == {"growth", "underflow", "overflow"}:
            ax = ca.variable_uoflow_growth(edges)
        elif options == {"underflow", "overflow"}:
            ax = ca.variable_uoflow(edges)
        elif options == {"underflow"}:
            ax = ca.variable_uflow(edges)
        elif options == {"overflow"}:
            ax = ca.variable_oflow(edges)
        elif options in (
            {"circular", "underflow", "overflow"},
            {"circular", "overflow"},
        ):
            # growth=True, underflow=False is also correct
            ax = ca.variable_circular(edges)
        elif options == set():
            ax = ca.variable_none(edges)
        else:
            raise KeyError("Unsupported collection of options")

        super().__init__(ax, metadata, __dict__)

    def _repr_args_(self) -> list[str]:
        "Return inner part of signature for use in repr"

        if len(self) > 20:
            ret = [repr(self.edges)]
        else:
            args = ", ".join(format(v, "g") for v in self.edges)
            ret = [f"[{args}]"]

        if self.traits.growth:
            ret.append("growth=True")
        elif self.traits.circular:
            ret.append("circular=True")
        else:
            if not self.traits.underflow:
                ret.append("underflow=False")
            if not self.traits.overflow:
                ret.append("overflow=False")

        ret += super()._repr_args_()

        return ret


@register(
    {
        ca.integer_none,
        ca.integer_uflow,
        ca.integer_oflow,
        ca.integer_uoflow,
        ca.integer_growth,
        ca.integer_circular,
    }
)
class Integer(Axis, family=boost_histogram):
    __slots__ = ()

    def __init__(
        self,
        start: int,
        stop: int,
        *,
        metadata: Any = None,
        underflow: bool = True,
        overflow: bool = True,
        growth: bool = False,
        circular: bool = False,
        __dict__: dict[str, Any] | None = None,
    ):
        """
        Make an integer axis, with a collection of consecutive integers.

        Parameters
        ----------
        start : int
            The beginning value for the axis
        stop : int
            The ending value for the axis. (start-stop) bins will be created.
        metadata : object
            Any Python object to attach to the axis, like a label.
        underflow : bool = True
            Enable the underflow bin
        overflow : bool = True
            Enable the overflow bin
        circular : bool = False
            Enable wraparound
        growth : bool = False
            Allow the axis to grow if a value is encountered out of range.
            Be careful, the axis will grow as large as needed.
        __dict__: Optional[dict[str, Any]] = None
            The full metadata dictionary
        """

        options = _opts(
            underflow=underflow, overflow=overflow, growth=growth, circular=circular
        )

        ax: ca._BaseInteger

        # underflow and overflow settings are ignored, integers are always
        # finite and thus cannot end up in a flow bin when growth is on
        if "growth" in options and "circular" not in options:
            ax = ca.integer_growth(start, stop)
        elif options == {"underflow", "overflow"}:
            ax = ca.integer_uoflow(start, stop)
        elif options == {"underflow"}:
            ax = ca.integer_uflow(start, stop)
        elif options == {"overflow"}:
            ax = ca.integer_oflow(start, stop)
        elif "circular" in options and "growth" not in options:
            ax = ca.integer_circular(start, stop)
        elif options == set():
            ax = ca.integer_none(start, stop)
        else:
            raise KeyError("Unsupported collection of options")

        super().__init__(ax, metadata, __dict__)

    def _repr_args_(self) -> list[str]:
        "Return inner part of signature for use in repr"

        ret = [f"{self.edges[0]:g}", f"{self.edges[-1]:g}"]

        if self.traits.growth:
            ret.append("growth=True")
        elif self.traits.circular:
            ret.append("circular=True")
        else:
            if not self.traits.underflow:
                ret.append("underflow=False")
            if not self.traits.overflow:
                ret.append("overflow=False")

        ret += super()._repr_args_()

        return ret


class BaseCategory(Axis, family=boost_histogram):
    __slots__ = ()

    def _repr_args_(self) -> list[str]:
        "Return inner part of signature for use in repr"

        ret = []

        if self.traits.growth:
            ret.append("growth=True")
        elif not self.traits.overflow:
            ret.append("overflow=False")

        ret += super()._repr_args_()
        return ret


@register({ca.category_str_growth, ca.category_str, ca.category_str_none})
class StrCategory(BaseCategory, family=boost_histogram):
    __slots__ = ()

    def __init__(
        self,
        categories: Iterable[str],
        *,
        metadata: Any = None,
        growth: bool = False,
        overflow: bool = True,
        __dict__: dict[str, Any] | None = None,
    ):
        """
        Make a category axis with strings; items will
        be added to a predefined list of bins or a growing (with growth=True)
        list of bins.


        Parameters
        ----------
        categories : Iterator[str]
            The bin values in strings. May be empty if growth is enabled.
        metadata : object
            Any Python object to attach to the axis, like a label.
        growth : bool = False
            Allow the axis to grow if a value is encountered out of range.
            Be careful, the axis will grow as large as needed.
        overflow : bool = True
            Include an overflow bin for "missed" hits. Ignored if growth=True.
        __dict__: Optional[dict[str, Any]] = None
            The full metadata dictionary
        """

        options = _opts(growth=growth, overflow=overflow)

        ax: ca._BaseCatStr

        # henryiii: We currently expand "abc" to "a", "b", "c" - some
        # Python interfaces protect against that

        if "growth" in options:
            ax = ca.category_str_growth(tuple(categories))
        elif options == {"overflow"}:
            ax = ca.category_str(tuple(categories))
        elif not options:
            ax = ca.category_str_none(tuple(categories))
        else:
            raise KeyError("Unsupported collection of options")

        super().__init__(ax, metadata, __dict__)

    def index(self, value: float | str) -> int:
        """
        Return the fractional index(es) given a value (or values) on the axis.
        """

        if _isstr(value):
            return self._ax.index(value)  # type: ignore[no-any-return]

        msg = f"index({value}) must be a string or iterable of strings for a StrCategory axis"
        raise TypeError(msg)

    def _repr_args_(self) -> list[str]:
        "Return inner part of signature for use in repr"

        args = ", ".join(repr(c) for c in self)
        ret = [f"[{args}]"]
        ret += super()._repr_args_()
        return ret


@register({ca.category_int, ca.category_int_growth, ca.category_int_none})
class IntCategory(BaseCategory, family=boost_histogram):
    __slots__ = ()

    def __init__(
        self,
        categories: Iterable[int],
        *,
        metadata: Any = None,
        growth: bool = False,
        overflow: bool = True,
        __dict__: dict[str, Any] | None = None,
    ):
        """
        Make a category axis with ints; items will
        be added to a predefined list of bins or a growing (with growth=True)
        list of bins. An empty list is allowed if growth=True.


        Parameters
        ----------
        categories : Iterable[int]
            The bin values, either ints or strings.
        metadata : object
            Any Python object to attach to the axis, like a label.
        growth : bool = False
            Allow the axis to grow if a value is encountered out of range.
            Be careful, the axis will grow as large as needed.
        overflow : bool = True
            Include an overflow bin for "missed" hits. Ignored if growth=True.
        __dict__: Optional[dict[str, Any]] = None
            The full metadata dictionary
        """

        options = _opts(growth=growth, overflow=overflow)
        ax: ca._BaseCatInt

        if "growth" in options:
            ax = ca.category_int_growth(tuple(categories))
        elif options == {"overflow"}:
            ax = ca.category_int(tuple(categories))
        elif not options:
            ax = ca.category_int_none(tuple(categories))
        else:
            raise KeyError("Unsupported collection of options")

        super().__init__(ax, metadata, __dict__)

    def _repr_args_(self) -> list[str]:
        "Return inner part of signature for use in repr"

        args = ", ".join(format(c, "g") for c in self)
        ret = [f"[{args}]"]
        ret += super()._repr_args_()
        return ret


# Contains all common methods and properties for the boolean axis
@register({ca.boolean})
class Boolean(Axis, family=boost_histogram):
    __slots__ = ()

    def __init__(self, *, metadata: Any = None, __dict__: dict[str, Any] | None = None):
        """
        Make an axis for boolean values.

        Parameters
        ----------
        metadata : object
            Any Python object to attach to the axis, like a label.
        __dict__: Optional[dict[str, Any]] = None
            The full metadata dictionary
        """

        ax = ca.boolean()

        super().__init__(ax, metadata, __dict__)

    def _repr_args_(self) -> list[str]:
        "Return inner part of signature for use in repr"
        ret = []

        if self.size == 0:
            ret.append("<empty>")
        elif self.size == 1 and self.centers[0] < 0.75:
            ret.append("<False>")
        elif self.size == 1:
            ret.append("<True>")

        ret += super()._repr_args_()
        return ret


class MGridOpts(TypedDict):
    sparse: bool
    indexing: Literal["ij", "xy"]


A = TypeVar("A", bound="ArrayTuple")


class ArrayTuple(tuple):  # type: ignore[type-arg]
    __slots__ = ()
    # This is an exhaustive list as of NumPy 1.19
    _REDUCTIONS = frozenset(("sum", "any", "all", "min", "max", "prod"))

    def __getattr__(self, name: str) -> Any:
        if name in self._REDUCTIONS:
            return partial(getattr(np, name), np.broadcast_arrays(*self))

        return self.__class__(getattr(a, name) for a in self)

    def __dir__(self) -> list[str]:
        names = dir(self.__class__) + dir("np.typing.NDArray[Any]")
        return sorted(n for n in names if not n.startswith("_"))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.__class__(a(*args, **kwargs) for a in self)

    def broadcast(self) -> Self:
        """
        The arrays in this tuple will be compressed if possible to save memory.
        Use this method to broadcast them out into their full memory
        representation.
        """
        return self.__class__(np.broadcast_arrays(*self))


B = TypeVar("B", bound="AxesTuple")


class AxesTuple(tuple):  # type: ignore[type-arg]
    __slots__ = ()
    _MGRIDOPTS: ClassVar[MGridOpts] = {"sparse": True, "indexing": "ij"}

    def __init__(self, /, _iterable: Iterable[Axis]) -> None:
        for item in self:
            if not isinstance(item, Axis):
                raise TypeError(
                    f"Only an iterable of Axis supported in AxesTuple, got {item}"
                )
        super().__init__()

    @property
    def size(self) -> tuple[int, ...]:
        return tuple(s.size for s in self)

    @property
    def extent(self) -> tuple[int, ...]:
        return tuple(s.extent for s in self)

    @property
    def centers(self) -> ArrayTuple:
        gen = (s.centers for s in self)
        return ArrayTuple(np.meshgrid(*gen, **self._MGRIDOPTS))

    @property
    def edges(self) -> ArrayTuple:
        gen = (s.edges for s in self)
        return ArrayTuple(np.meshgrid(*gen, **self._MGRIDOPTS))

    @property
    def widths(self) -> ArrayTuple:
        gen = (s.widths for s in self)
        return ArrayTuple(np.meshgrid(*gen, **self._MGRIDOPTS))

    def value(self, *indexes: float) -> tuple[float, ...]:
        if len(indexes) != len(self):
            raise IndexError(
                "Must have the same number of arguments as the number of axes"
            )
        return tuple(self[i].value(indexes[i]) for i in range(len(indexes)))

    def bin(self, *indexes: float) -> tuple[float, ...]:
        if len(indexes) != len(self):
            raise IndexError(
                "Must have the same number of arguments as the number of axes"
            )
        return tuple(self[i].bin(indexes[i]) for i in range(len(indexes)))

    def index(self, *values: float) -> tuple[float, ...]:  # type: ignore[override, override]
        if len(values) != len(self):
            raise IndexError(
                "Must have the same number of arguments as the number of axes"
            )
        return tuple(self[i].index(values[i]) for i in range(len(values)))

    def __getitem__(self, item: Any) -> Any:
        result = super().__getitem__(item)
        return self.__class__(result) if isinstance(result, tuple) else result

    def __getattr__(self, attr: str) -> tuple[Any, ...]:
        return tuple(getattr(s, attr) for s in self)

    def __setattr__(self, attr: str, values: Any) -> None:
        try:
            super().__setattr__(attr, values)
        except AttributeError:
            for s, v in zip_strict(self, values):
                s.__setattr__(attr, v)

    value.__doc__ = Axis.value.__doc__
    index.__doc__ = Axis.index.__doc__
    bin.__doc__ = Axis.bin.__doc__
