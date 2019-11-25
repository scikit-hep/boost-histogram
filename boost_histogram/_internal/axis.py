from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from .._core.axis import options

from .._core import axis as ca

from .kwargs import KWArgs
from .sig_tools import inject_signature
from .axis_transform import AxisTransform
from .utils import cast, register, set_family, MAIN_FAMILY, CPP_FAMILY, set_module

import warnings
import copy


# Contains common methods and properties to all axes
@set_module("boost_histogram.axis")
class Axis(object):
    __slots__ = ("_ax",)

    def __copy__(self):
        other = self.__class__.__new__(self.__class__)
        other._ax = copy.copy(self._ax)
        return other

    def index(self, value):
        """
        Return the fractional index(es) given a value (or values) on the axis.
        """

        return self._ax.index(value)

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

    @property
    def metadata(self):
        """
        Get or set the metadata associated with this axis.
        """
        return self._ax.metadata

    @metadata.setter
    def metadata(self, value):
        self._ax.metadata = value

    @classmethod
    def _convert_cpp(cls, cpp_object):
        nice_ax = cls.__new__(cls)
        nice_ax._ax = cpp_object
        return nice_ax

    def __len__(self):
        return self._ax.size

    def __iter__(self):
        return self._ax.__iter__()


# Mixin for main style classes
# Contains common methods and properties to all Main module axes
class MainAxisMixin(object):
    __slots__ = ()

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

        if self.metadata is not None:
            ret += ", metadata={0!r}".format(self.metadata)

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
        if i < 0:
            i += self._ax.size
        if i >= self._ax.size:
            raise IndexError("Out of range access")
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


# Contains all common methods for cpp module axes
class CppAxisMixin(object):
    __slots__ = ()

    def __repr__(self):
        return repr(self._ax)

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

    def size(self):
        """
        Return number of bins excluding under- and overflow.
        """
        return self._ax.size

    def extent(self):
        """
        Return number of bins including under- and overflow.
        """
        return self._ax.extent

    def edges(self):
        return self._ax.edges

    def centers(self):
        """
        An array of bin centers.
        """
        return self._ax.centers

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
class BaseRegular(Axis):
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
            Any Python object to attach to the axis, like a label.
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

            self._ax = transform._produce(bins, start, stop, metadata)

        elif options == {"growth", "underflow", "overflow"}:
            self._ax = ca.regular_uoflow_growth(bins, start, stop, metadata)
        elif options == {"underflow", "overflow"}:
            self._ax = ca.regular_uoflow(bins, start, stop, metadata)
        elif options == {"underflow"}:
            self._ax = ca.regular_uflow(bins, start, stop, metadata)
        elif options == {"overflow"}:
            self._ax = ca.regular_oflow(bins, start, stop, metadata)
        elif options == {
            "circular",
            "underflow",
            "overflow",
        } or options == {  # growth=True should work
            "circular",
            "overflow",
        }:  # growth=True, underflow=False is also correct
            self._ax = ca.regular_circular(bins, start, stop, metadata)

        elif options == set():
            self._ax = ca.regular_none(bins, start, stop, metadata)
        else:
            raise KeyError("Unsupported collection of options")


@set_module("boost_histogram.axis")
@set_family(MAIN_FAMILY)
class Regular(BaseRegular, MainAxisMixin):
    __slots__ = ()

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


@set_module("boost_histogram.cpp.axis")
@set_family(MAIN_FAMILY)
class regular(BaseRegular, CppAxisMixin):
    __slots__ = ()

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
class BaseVariable(Axis):
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
            self._ax = ca.variable_uoflow_growth(edges, metadata)
        elif options == {"underflow", "overflow"}:
            self._ax = ca.variable_uoflow(edges, metadata)
        elif options == {"underflow"}:
            self._ax = ca.variable_uflow(edges, metadata)
        elif options == {"overflow"}:
            self._ax = ca.variable_oflow(edges, metadata)
        elif options == {
            "circular",
            "underflow",
            "overflow",
        } or options == {  # growth=True should work
            "circular",
            "overflow",
        }:  # growth=True, underflow=False is also correct
            self._ax = ca.variable_circular(edges, metadata)
        elif options == set():
            self._ax = ca.variable_none(edges, metadata)
        else:
            raise KeyError("Unsupported collection of options")


@set_family(MAIN_FAMILY)
@set_module("boost_histogram.axis")
class Variable(BaseVariable, MainAxisMixin):
    __slots__ = ()

    def _repr_args(self):
        "Return inner part of signature for use in repr"

        if len(self) > 20:
            return repr(self.edges)
        else:
            return "[{}]".format(", ".join(format(v, "g") for v in self.edges))


@set_family(CPP_FAMILY)
@set_module("boost_histogram.cpp.axis")
class variable(BaseVariable, CppAxisMixin):
    __slots__ = ()


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
class BaseInteger(Axis):
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
            self._ax = ca.integer_growth(start, stop, metadata)
        elif options == {"underflow", "overflow"}:
            self._ax = ca.integer_uoflow(start, stop, metadata)
        elif options == {"underflow"}:
            self._ax = ca.integer_uflow(start, stop, metadata)
        elif options == {"overflow"}:
            self._ax = ca.integer_oflow(start, stop, metadata)
        elif (
            "circular" in options and "growth" not in options
        ):  # growth=True should work
            self._ax = ca.integer_circular(
                start, stop, metadata
            )  # flow bins do no matter
        elif options == set():
            self._ax = ca.integer_none(start, stop, metadata)
        else:
            raise KeyError("Unsupported collection of options")


@set_family(MAIN_FAMILY)
@set_module("boost_histogram.axis")
class Integer(BaseInteger, MainAxisMixin):
    __slots__ = ()

    def _repr_args(self):
        "Return inner part of signature for use in repr"

        return "{start:g}, {stop:g}".format(start=self.edges[0], stop=self.edges[-1])


@set_family(CPP_FAMILY)
@set_module("boost_histogram.cpp.axis")
class integer(BaseInteger, CppAxisMixin):
    __slots__ = ()


@register({ca.category_str_growth, ca.category_str})
class BaseStrCategory(Axis):
    __slots__ = ()

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

        # We need to make sure we support Python 2 for now :(
        # henryiii: This shortcut possibly should be removed
        if isinstance(categories, (type(""), type(u""))):
            categories = list(categories)

        if options == {"growth"}:
            self._ax = ca.category_str_growth(categories, metadata)
        elif options == set():
            self._ax = ca.category_str(categories, metadata)
        else:
            raise KeyError("Unsupported collection of options")


@register({ca.category_int, ca.category_int_growth})
class BaseIntCategory(Axis):
    __slots__ = ()

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
            self._ax = ca.category_int_growth(categories, metadata)
        elif options == set():
            self._ax = ca.category_int(categories, metadata)
        else:
            raise KeyError("Unsupported collection of options")


class CategoryMixin(object):
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

        if self.metadata is not None:
            ret += ", metadata={0!r}".format(self.metadata)

        return ret


@set_family(MAIN_FAMILY)
@set_module("boost_histogram.axis")
class StrCategory(BaseStrCategory, CategoryMixin, MainAxisMixin):
    __slots__ = ()

    def _repr_args(self):
        "Return inner part of signature for use in repr"

        return "[{0}]".format(", ".join(repr(c) for c in self))


@set_family(MAIN_FAMILY)
@set_module("boost_histogram.axis")
class IntCategory(BaseIntCategory, CategoryMixin, MainAxisMixin):
    __slots__ = ()

    def _repr_args(self):
        "Return inner part of signature for use in repr"

        return "[{0}]".format(", ".join(format(c, "g") for c in self))


@inject_signature("categories, *, metadata=None, growth=False")
def Category(categories, **kwargs):
    """
    Deprecated: Use IntCategory or StrCategory instead.
    This shortcut may return eventually.
    """
    warnings.warn("Use IntCategory or StrCategory instead of Category", FutureWarning)

    if len(categories) < 1:
        raise TypeError(
            "Cannot deduce int vs. str, please use IntCategory/StrCategory instead"
        )

    try:
        return IntCategory(categories, **kwargs)
    except TypeError:
        return StrCategory(categories, **kwargs)


@set_family(CPP_FAMILY)
@set_module("boost_histogram.cpp.axis")
class int_category(BaseIntCategory, CppAxisMixin):
    __slots__ = ()


@set_family(CPP_FAMILY)
@set_module("boost_histogram.cpp.axis")
class str_category(BaseStrCategory, CppAxisMixin):
    __slots__ = ()
