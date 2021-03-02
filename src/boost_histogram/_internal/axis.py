import copy
from typing import Any, Dict, Optional, Tuple, Union

from .._core import axis as ca
from .axis_transform import AxisTransform
from .traits import Traits
from .utils import MAIN_FAMILY, cast, register, set_family, set_module


def _isstr(value: Any) -> bool:
    """
    Check to see if this is a stringlike or a (nested) iterable of stringlikes
    """

    if isinstance(value, (str, bytes)):
        return True
    elif hasattr(value, "__iter__"):
        return all(_isstr(v) for v in value)
    else:
        return False


def opts(**kwargs: bool):
    return {k for k, v in kwargs.items() if v}


# Contains common methods and properties to all axes
@set_module("boost_histogram.axis")
class Axis:
    __slots__ = ("_ax", "__dict__")

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr == "__dict__":
            self._ax.metadata = value
        object.__setattr__(self, attr, value)

    def __getattr__(self, attr: str) -> Any:
        if attr == "metadata":
            return None
        raise AttributeError(
            f"object {self.__class__.__name__} has not attribute {attr}"
        )

    def __init__(
        self, ax, metadata: Optional[Dict[str, Any]], __dict__: Optional[Dict[str, Any]]
    ):
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
        elif __dict__ is not None:
            self._ax.metadata = __dict__
        elif metadata is not None:
            self._ax.metadata["metadata"] = metadata

        self.__dict__ = self._ax.metadata

    def __setstate__(self, state):
        self._ax = state["_ax"]
        self.__dict__ = self._ax.metadata

    def __getstate__(self):
        return {"_ax": self._ax}

    def __copy__(self):
        other = self.__class__.__new__(self.__class__)
        other._ax = copy.copy(self._ax)
        other.__dict__ = other._ax.metadata
        return other

    def index(self, value: float) -> int:
        """
        Return the fractional index(es) given a value (or values) on the axis.
        """

        if not _isstr(value):
            return self._ax.index(value)
        else:
            raise TypeError(
                "index({value}) cannot be a string for a numerical axis".format(
                    value=value
                )
            )

    def value(self, index: float) -> float:
        """
        Return the value(s) given an (fractional) index (or indices).
        """

        return self._ax.value(index)

    def bin(self, index: float) -> Union[float, str, Tuple[float, float]]:
        """
        Return the edges of the bins as a tuple for a
        continuous axis or the bin value for a
        non-continuous axis, when given an index.
        """

        return self._ax.bin(index)

    def __eq__(self, other):
        return self._ax == other._ax

    def __ne__(self, other):
        return self._ax != other._ax

    @classmethod
    def _convert_cpp(cls, cpp_object):
        nice_ax = cls.__new__(cls)
        nice_ax._ax = cpp_object
        nice_ax.__dict__ = cpp_object.metadata
        return nice_ax

    def __len__(self) -> int:
        return self._ax.size

    def __iter__(self):
        return self._ax.__iter__()

    def _process_loc(self, start, stop) -> Tuple[int, int]:
        """
        Compute start and stop into actual start and stop values in Boost.Histogram.
        None -> -1 or 0 for start, -> len or len+1 for stop. If start or stop are
        callable, then call them with the axes.
        """

        def _process_internal(item, default):
            return default if item is None else item(self) if callable(item) else item

        begin = _process_internal(start, -1 if self._ax.traits_underflow else 0)
        end = _process_internal(
            stop, len(self) + (1 if self._ax.traits_overflow else 0)
        )

        return begin, end

    def __repr__(self) -> str:
        return "{self.__class__.__name__}({args}{kwargs})".format(
            self=self, args=self._repr_args(), kwargs=self._repr_kwargs()
        )

    def _repr_kwargs(self) -> str:
        """
        Return options for use in repr. Metadata is last,
        just in case it spans multiple lines.
        """

        ret = ""
        if self.traits.growth:
            ret += ", growth=True"
        elif self.traits.circular:
            ret += ", circular=True"
        else:
            if not self.traits.underflow:
                ret += ", underflow=False"
            if not self.traits.overflow:
                ret += ", overflow=False"

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
        return self._ax.size

    @property
    def extent(self) -> int:
        """
        Return number of bins including under- and overflow.
        """
        return self._ax.extent

    def __getitem__(self, i):
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
        return self.bin(i)

    @property
    def edges(self):
        return self._ax.edges

    @property
    def centers(self):
        """
        An array of bin centers.
        """
        return self._ax.centers

    @property
    def widths(self):
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
        ca.regular_numpy,
        ca.regular_pow,
        ca.regular_trans,
        ca.regular_circular,
    }
)
@set_module("boost_histogram.axis")
@set_family(MAIN_FAMILY)
class Regular(Axis):
    __slots__ = ()

    def __init__(
        self,
        bins,
        start,
        stop,
        *,
        metadata=None,
        underflow=True,
        overflow=True,
        growth=False,
        circular=False,
        transform=None,
        __dict__=None,
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
        __dict__: Optional[Dict[str, Any]] = None
            The full metadata dictionary
        """

        options = opts(
            underflow=underflow, overflow=overflow, growth=growth, circular=circular
        )

        if transform is not None:
            if options != {"underflow", "overflow"}:
                raise KeyError("Transform supplied, cannot change other options")

            if (
                not isinstance(transform, AxisTransform)
                and AxisTransform in transform.__bases__
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
        elif options == {"circular", "underflow", "overflow"} or options == {
            "circular",
            "overflow",
        }:
            # growth=True, underflow=False is also correct
            ax = ca.regular_circular(bins, start, stop)

        elif options == set():
            ax = ca.regular_none(bins, start, stop)
        else:
            raise KeyError("Unsupported collection of options")

        super().__init__(ax, metadata, __dict__)

    def _repr_args(self) -> str:
        "Return inner part of signature for use in repr"

        return "{bins:g}, {start:g}, {stop:g}".format(
            bins=self.size, start=self.edges[0], stop=self.edges[-1]
        )

    def _repr_kwargs(self) -> str:
        ret = super()._repr_kwargs()

        if self.transform is not None:
            ret += f", transform={self.transform}"

        return ret

    @property
    def transform(self) -> Optional[AxisTransform]:
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
@set_family(MAIN_FAMILY)
@set_module("boost_histogram.axis")
class Variable(Axis):
    __slots__ = ()

    def __init__(
        self,
        edges,
        *,
        metadata: Any = None,
        underflow: bool = True,
        overflow: bool = True,
        growth: bool = False,
        circular: bool = False,
        __dict__: Dict[str, Any] = None,
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
        __dict__: Optional[Dict[str, Any]] = None
            The full metadata dictionary
        """

        options = opts(
            underflow=underflow, overflow=overflow, growth=growth, circular=circular
        )

        if options == {"growth", "underflow", "overflow"}:
            ax = ca.variable_uoflow_growth(edges)
        elif options == {"underflow", "overflow"}:
            ax = ca.variable_uoflow(edges)
        elif options == {"underflow"}:
            ax = ca.variable_uflow(edges)
        elif options == {"overflow"}:
            ax = ca.variable_oflow(edges)
        elif options == {"circular", "underflow", "overflow",} or options == {
            "circular",
            "overflow",
        }:
            # growth=True, underflow=False is also correct
            ax = ca.variable_circular(edges)
        elif options == set():
            ax = ca.variable_none(edges)
        else:
            raise KeyError("Unsupported collection of options")

        super().__init__(ax, metadata, __dict__)

    def _repr_args(self) -> str:
        "Return inner part of signature for use in repr"

        if len(self) > 20:
            return repr(self.edges)
        else:
            return "[{}]".format(", ".join(format(v, "g") for v in self.edges))


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
@set_family(MAIN_FAMILY)
@set_module("boost_histogram.axis")
class Integer(Axis):
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
        __dict__: Dict[str, Any] = None,
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
        __dict__: Optional[Dict[str, Any]] = None
            The full metadata dictionary
        """

        options = opts(
            underflow=underflow, overflow=overflow, growth=growth, circular=circular
        )

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

    def _repr_args(self):
        "Return inner part of signature for use in repr"

        return "{start:g}, {stop:g}".format(start=self.edges[0], stop=self.edges[-1])


class BaseCategory(Axis):
    __slots__ = ()

    def _repr_kwargs(self):
        """
        Return options for use in repr. Metadata is last,
        just in case it spans multiple lines.


        This is specialized for Category axes to avoid repeating
        the flow arguments unnecessarily.
        """

        ret = ""
        if self.traits.growth:
            ret += ", growth=True"
        elif self.traits.circular:
            ret += ", circular=True"

        return ret


@set_family(MAIN_FAMILY)
@set_module("boost_histogram.axis")
@register({ca.category_str_growth, ca.category_str})
class StrCategory(BaseCategory):
    __slots__ = ()

    def __init__(
        self,
        categories,
        *,
        metadata: Any = None,
        growth: bool = False,
        __dict__: Dict[str, Any] = None,
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
        __dict__: Optional[Dict[str, Any]] = None
            The full metadata dictionary
        """

        options = opts(growth=growth)

        # henryiii: We currently expand "abc" to "a", "b", "c" - some
        # Python interfaces protect against that

        if options == {"growth"}:
            ax = ca.category_str_growth(tuple(categories))
        elif options == set():
            ax = ca.category_str(tuple(categories))
        else:
            raise KeyError("Unsupported collection of options")

        super().__init__(ax, metadata, __dict__)

    def index(self, value):
        """
        Return the fractional index(es) given a value (or values) on the axis.
        """

        if _isstr(value):
            return self._ax.index(value)
        else:
            raise TypeError(
                "index({value}) must be a string or iterable of strings for a StrCategory axis".format(
                    value=value
                )
            )

    def _repr_args(self):
        "Return inner part of signature for use in repr"

        return "[{}]".format(", ".join(repr(c) for c in self))


@set_family(MAIN_FAMILY)
@set_module("boost_histogram.axis")
@register({ca.category_int, ca.category_int_growth})
class IntCategory(BaseCategory):
    __slots__ = ()

    def __init__(
        self,
        categories,
        *,
        metadata: Any = None,
        growth: bool = False,
        __dict__: Optional[Dict[str, Any]] = None,
    ):
        """
        Make a category axis with ints; items will
        be added to a predefined list of bins or a growing (with growth=True)
        list of bins. An empty list is allowed if growth=True.


        Parameters
        ----------
        categories : Iteratable[int]
            The bin values, either ints or strings.
        metadata : object
            Any Python object to attach to the axis, like a label.
        growth : bool = False
            Allow the axis to grow if a value is encountered out of range.
            Be careful, the axis will grow as large as needed.
        __dict__: Optional[Dict[str, Any]] = None
            The full metadata dictionary
        """

        options = opts(growth=growth)

        if options == {"growth"}:
            ax = ca.category_int_growth(tuple(categories))
        elif options == set():
            ax = ca.category_int(tuple(categories))
        else:
            raise KeyError("Unsupported collection of options")

        super().__init__(ax, metadata, __dict__)

    def _repr_args(self) -> str:
        "Return inner part of signature for use in repr"

        return "[{}]".format(", ".join(format(c, "g") for c in self))


# Contains all common methods and properties for the boolean axis
@register({ca.boolean})
@set_family(MAIN_FAMILY)
@set_module("boost_histogram.axis")
class Boolean(Axis):
    __slots__ = ()

    def __init__(self, *, metadata: Any = None, __dict__: Dict[str, Any] = None):
        """
        Make an axis for boolean values.

        Parameters
        ----------
        metadata : object
            Any Python object to attach to the axis, like a label.
        __dict__: Optional[Dict[str, Any]] = None
            The full metadata dictionary
        """

        ax = ca.boolean()

        super().__init__(ax, metadata, __dict__)

    def _repr_args(self) -> str:
        "Return inner part of signature for use in repr"
        if self.size == 2:
            return ""
        elif self.size == 0:
            return "<empty>"
        elif self.size == 1 and self.centers[0] < 0.75:
            return "<False>"
        elif self.size == 1:
            return "<True>"
        else:
            # Shouldn't be possible, can't grow
            return "<unknown>"

    def _repr_kwargs(self) -> str:
        """
        Return options for use in repr. Metadata is last,
        just in case it spans multiple lines.
        """

        return ""
