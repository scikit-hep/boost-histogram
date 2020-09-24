# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from .._core import axis as ca

from .kwargs import KWArgs
from .sig_tools import inject_signature
from .axis_transform import AxisTransform
from .utils import cast, register, set_family, MAIN_FAMILY, set_module
from .six import string_types

import copy

del absolute_import, division, print_function


def _isstr(value):
    """
    Check to see if this is a stringlike or a (nested) iterable of stringlikes
    """

    if isinstance(value, string_types + (bytes,)):
        return True
    elif hasattr(value, "__iter__"):
        return all(_isstr(v) for v in value)
    else:
        return False


# Contains common methods and properties to all axes
@set_module("boost_histogram.axis")
class Axis(object):
    __slots__ = ("_ax",)

    def __copy__(self):
        other = self.__class__.__new__(self.__class__)
        other._ax = copy.copy(self._ax)
        return other

    def __getattr__(self, item):
        if item == "_ax":
            return Axis.__dict__[item].__get__(self)
        elif item in self._ax.metadata:
            return self._ax.metadata[item]
        elif item == "metadata":
            return None
        else:
            msg = "'{}' object has no attribute '{}' in {}".format(
                type(self).__name__, item, set(self._ax.metadata)
            )
            raise AttributeError(msg)

    def __setattr__(self, item, value):
        if item == "_ax":
            Axis.__dict__[item].__set__(self, value)
        else:
            self._ax.metadata[item] = value

    def __dir__(self):
        metadata = list(self._ax.metadata)
        return sorted(dir(type(self)) + metadata)

    def index(self, value):
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

    def value(self, index):
        """
        Return the value(s) given an (fractional) index (or indices).
        """

        return self._ax.value(index)

    def bin(self, index):
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
        return nice_ax

    def __len__(self):
        return self._ax.size

    def __iter__(self):
        return self._ax.__iter__()

    def _process_loc(self, start, stop):
        """
        Compute start and stop into actual start and stop values in Boost.Histogram.
        None -> -1 or 0 for start, -> len or len+1 for stop. If start or stop are
        callable, then call them with the axes.
        """

        def _process_internal(item, default):
            return default if item is None else item(self) if callable(item) else item

        begin = _process_internal(start, -1 if self._ax.options.underflow else 0)
        end = _process_internal(
            stop, len(self) + (1 if self._ax.options.overflow else 0)
        )

        return begin, end

    def __repr__(self):
        return "{self.__class__.__name__}({args}{kwargs})".format(
            self=self, args=self._repr_args(), kwargs=self._repr_kwargs()
        )

    def _repr_kwargs(self):
        """
        Return options for use in repr. Metadata is last,
        just in case it spans multiple lines.
        """

        ret = ""
        if self.options.growth:
            ret += ", growth=True"
        elif self.options.circular:
            ret += ", circular=True"
        else:
            if not self.options.underflow:
                ret += ", underflow=False"
            if not self.options.overflow:
                ret += ", overflow=False"

        return ret

    @property
    def options(self):
        """
        Return the options.  Fields:
          .underflow - True if axes captures values that are too small
          .overflow  - True if axes captures values that are too large
                       (or non-valid for category axes)
          .growth    - True if axis can grow
          .circular  - True if axis wraps around
        """
        return self._ax.options

    @property
    def size(self):
        """
        Return number of bins excluding under- and overflow.
        """
        return self._ax.size

    @property
    def extent(self):
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
                    "Out of range access, {0} is more than {1}".format(i, self._ax.size)
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

    @inject_signature(
        "self, bins, start, stop, *, metadata=None, underflow=True, overflow=True, growth=False, circular=False, transform=None"
    )
    def __init__(self, bins, start, stop, **kwargs):
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
        """

        with KWArgs(kwargs) as k:
            metadata = k.optional("metadata")
            transform = k.optional("transform")
            options = k.options(
                underflow=True, overflow=True, growth=False, circular=False
            )

        if transform is not None:
            if options != {"underflow", "overflow"}:
                raise KeyError("Transform supplied, cannot change other options")

            if (
                not isinstance(transform, AxisTransform)
                and AxisTransform in transform.__bases__
            ):
                raise TypeError("You must pass an instance, use {}()".format(transform))

            self._ax = transform._produce(bins, start, stop)

        elif options == {"growth", "underflow", "overflow"}:
            self._ax = ca.regular_uoflow_growth(bins, start, stop)
        elif options == {"underflow", "overflow"}:
            self._ax = ca.regular_uoflow(bins, start, stop)
        elif options == {"underflow"}:
            self._ax = ca.regular_uflow(bins, start, stop)
        elif options == {"overflow"}:
            self._ax = ca.regular_oflow(bins, start, stop)
        elif options == {"circular", "underflow", "overflow"} or options == {
            "circular",
            "overflow",
        }:
            # growth=True, underflow=False is also correct
            self._ax = ca.regular_circular(bins, start, stop)

        elif options == set():
            self._ax = ca.regular_none(bins, start, stop)
        else:
            raise KeyError("Unsupported collection of options")

        self.metadata = metadata

    def _repr_args(self):
        "Return inner part of signature for use in repr"

        return "{bins:g}, {start:g}, {stop:g}".format(
            bins=self.size, start=self.edges[0], stop=self.edges[-1]
        )

    def _repr_kwargs(self):
        ret = super(Regular, self)._repr_kwargs()

        if self.transform is not None:
            ret += ", transform={0}".format(self.transform)

        return ret

    @property
    def transform(self):
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

    @inject_signature(
        "self, edges, *, metadata=None, underflow=True, overflow=True, growth=False"
    )
    def __init__(self, edges, **kwargs):
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
        """
        with KWArgs(kwargs) as k:
            metadata = k.optional("metadata")
            options = k.options(
                underflow=True, overflow=True, circular=False, growth=False
            )

        if options == {"growth", "underflow", "overflow"}:
            self._ax = ca.variable_uoflow_growth(edges)
        elif options == {"underflow", "overflow"}:
            self._ax = ca.variable_uoflow(edges)
        elif options == {"underflow"}:
            self._ax = ca.variable_uflow(edges)
        elif options == {"overflow"}:
            self._ax = ca.variable_oflow(edges)
        elif options == {"circular", "underflow", "overflow",} or options == {
            "circular",
            "overflow",
        }:
            # growth=True, underflow=False is also correct
            self._ax = ca.variable_circular(edges)
        elif options == set():
            self._ax = ca.variable_none(edges)
        else:
            raise KeyError("Unsupported collection of options")

        self.metadata = metadata

    def _repr_args(self):
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

    @inject_signature(
        "self, start, stop, *, metadata=None, underflow=True, overflow=True, growth=False"
    )
    def __init__(self, start, stop, **kwargs):
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
        """
        with KWArgs(kwargs) as k:
            metadata = k.optional("metadata")
            options = k.options(
                underflow=True, overflow=True, circular=False, growth=False
            )

        # underflow and overflow settings are ignored, integers are always
        # finite and thus cannot end up in a flow bin when growth is on
        if "growth" in options and "circular" not in options:
            self._ax = ca.integer_growth(start, stop)
        elif options == {"underflow", "overflow"}:
            self._ax = ca.integer_uoflow(start, stop)
        elif options == {"underflow"}:
            self._ax = ca.integer_uflow(start, stop)
        elif options == {"overflow"}:
            self._ax = ca.integer_oflow(start, stop)
        elif "circular" in options and "growth" not in options:
            self._ax = ca.integer_circular(start, stop)
        elif options == set():
            self._ax = ca.integer_none(start, stop)
        else:
            raise KeyError("Unsupported collection of options")

        self.metadata = metadata

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
        if self.options.growth:
            ret += ", growth=True"
        elif self.options.circular:
            ret += ", circular=True"

        return ret


@set_family(MAIN_FAMILY)
@set_module("boost_histogram.axis")
@register({ca.category_str_growth, ca.category_str})
class StrCategory(BaseCategory):
    @inject_signature("self, categories, *, metadata=None, growth=False")
    def __init__(self, categories, **kwargs):
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
        """
        with KWArgs(kwargs) as k:
            metadata = k.optional("metadata")
            options = k.options(growth=False)

        # henryiii: We currently expand "abc" to "a", "b", "c" - some
        # Python interfaces protect against that

        if options == {"growth"}:
            self._ax = ca.category_str_growth(tuple(categories))
        elif options == set():
            self._ax = ca.category_str(tuple(categories))
        else:
            raise KeyError("Unsupported collection of options")

        self.metadata = metadata

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

        return "[{0}]".format(", ".join(repr(c) for c in self))


@set_family(MAIN_FAMILY)
@set_module("boost_histogram.axis")
@register({ca.category_int, ca.category_int_growth})
class IntCategory(BaseCategory):
    @inject_signature("self, categories, *, metadata=None, growth=False")
    def __init__(self, categories, **kwargs):
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
        """
        with KWArgs(kwargs) as k:
            metadata = k.optional("metadata")
            options = k.options(growth=False)

        if options == {"growth"}:
            self._ax = ca.category_int_growth(tuple(categories))
        elif options == set():
            self._ax = ca.category_int(tuple(categories))
        else:
            raise KeyError("Unsupported collection of options")

        self.metadata = metadata

    def _repr_args(self):
        "Return inner part of signature for use in repr"

        return "[{0}]".format(", ".join(format(c, "g") for c in self))


# Contains all common methods and properties for the boolean axis
@register({ca.boolean})
@set_family(MAIN_FAMILY)
@set_module("boost_histogram.axis")
class Boolean(Axis):
    __slots__ = ()

    @inject_signature("self, *, metadata=None")
    def __init__(self, **kwargs):
        """
        Make an axis for boolean values.

        Parameters
        ----------
        metadata : object
            Any Python object to attach to the axis, like a label.
        """
        with KWArgs(kwargs) as k:
            metadata = k.optional("metadata")

        self._ax = ca.boolean()
        self.metadata = metadata

    def _repr_args(self):
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
            "<unknown>"

    def _repr_kwargs(self):
        """
        Return options for use in repr. Metadata is last,
        just in case it spans multiple lines.
        """

        return ""
