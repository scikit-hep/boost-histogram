from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from .._core.axis import options

from .._core import axis as ca

from .kwargs import KWArgs
from .sig_tools import inject_signature
from .axis_transform import AxisTransform, _to_transform
from .utils import cast, register


class Axis(object):
    __slots__ = ("_ax",)

    def index(self, x):
        """
        Return the fractional index(es) given a value (or values) on the axis.
        """

        return self._ax.index(x)

    def value(self, i):
        """
        Return the value(s) given an (fractional) index (or indices).
        """

        return self._ax.value(i)

    def bin(self, i):
        """
        Return the edges of the bins as a tuple for a
        continuous axis or the bin value for a
        non-continuous axis, when given an index.
        """

        return self._ax.bin(i)

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

    def __eq__(self, other):
        return self._ax == other._ax

    def __ne__(self, other):
        return self._ax != other._ax

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
    def metadata(self):
        """
        Get or set the metadata associated with this axis.
        """
        return self._ax.metadata

    @metadata.setter
    def metadata(self, value):
        self._ax.metadata = value

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

    def __len__(self):
        return self._ax.size

    def __getitem__(self, i):
        """
        Access a bin, using normal Python syntax for wraparound.
        """
        if i < 0:
            i += self._ax.size
        if i >= self._ax.size:
            raise IndexError("Out of range access")
        return self.bin(i)

    def __iter__(self):
        return self._ax.__iter__()

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

    @classmethod
    def _convert_cpp(cls, cpp_object):
        nice_ax = cls.__new__(cls)
        nice_ax._ax = cpp_object
        return nice_ax


Axis.__module__ = "boost_histogram.axis"


@register(ca.regular_uoflow)
@register(ca.regular_uoflow_growth)
@register(ca.regular_uflow)
@register(ca.regular_oflow)
@register(ca.regular_none)
@register(ca.regular_numpy)
@register(ca.regular_sqrt)
@register(ca.regular_pow)
@register(ca.regular_log)
@register(ca.circular)
class Regular(Axis):
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
            return _to_transform(self._ax.transform)
        return None

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
        elif options == {"circular", "underflow", "overflow"}:
            self._ax = ca.circular(bins, start, stop, metadata)

        elif options == set():
            self._ax = ca.regular_none(bins, start, stop, metadata)
        else:
            raise KeyError("Unsupported collection of options")


Regular.__module__ = "boost_histogram.axis"


@register(ca.variable_none)
@register(ca.variable_uflow)
@register(ca.variable_oflow)
@register(ca.variable_uoflow)
@register(ca.variable_uoflow_growth)
class Variable(Axis):
    __slots__ = ()

    def _repr_args(self):
        "Return inner part of signature for use in repr"

        if len(self) > 20:
            return repr(self.edges)
        else:
            return "[{}]".format(", ".join(format(v, "g") for v in self.edges))

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
        growth : bool = False
            Allow the axis to grow if a value is encountered out of range.
            Be careful, the axis will grow as large as needed.
        """
        with KWArgs(kwargs) as k:
            metadata = k.optional("metadata")
            options = k.options(underflow=True, overflow=True, growth=False)

        if options == {"growth", "underflow", "overflow"}:
            self._ax = ca.variable_uoflow_growth(edges, metadata)
        elif options == {"underflow", "overflow"}:
            self._ax = ca.variable_uoflow(edges, metadata)
        elif options == {"underflow"}:
            self._ax = ca.variable_uflow(edges, metadata)
        elif options == {"overflow"}:
            self._ax = ca.variable_oflow(edges, metadata)
        elif options == set():
            self._ax = ca.variable_none(edges, metadata)
        else:
            raise KeyError("Unsupported collection of options")


Variable.__module__ = "boost_histogram.axis"


@register(ca.integer_none)
@register(ca.integer_uflow)
@register(ca.integer_oflow)
@register(ca.integer_uoflow)
@register(ca.integer_growth)
class Integer(Axis):
    __slots__ = ()

    def _repr_args(self):
        "Return inner part of signature for use in repr"

        return "{start:g}, {stop:g}".format(start=self.edges[0], stop=self.edges[-1])

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
        growth : bool = False
            Allow the axis to grow if a value is encountered out of range.
            Be careful, the axis will grow as large as needed.
        """
        with KWArgs(kwargs) as k:
            metadata = k.optional("metadata")
            options = k.options(underflow=True, overflow=True, growth=False)

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
        elif options == set():
            self._ax = ca.integer_none(start, stop, metadata)
        else:
            raise KeyError("Unsupported collection of options")


Integer.__module__ = "boost_histogram.axis"


@register(ca.category_int_growth)
@register(ca.category_str_growth)
@register(ca.category_int)
@register(ca.category_str)
class Category(Axis):
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

    def _repr_args(self):
        "Return inner part of signature for use in repr"

        return (
            "["
            + ", ".join(
                (repr(c) if isinstance(c, str) else format(c, "g")) for c in self
            )
            + "]"
        )

    @inject_signature("self, categories, *, metadata=None, growth=False")
    def __init__(self, categories, **kwargs):
        """
        Make a category axis with either ints or strings; items will
        be added to a predefined list of bins or a growing (with growth=True)
        list of bins.


        Parameters
        ----------
        categories : Union[Array[int], Array[str]]
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

        if isinstance(categories, str):
            categories = list(categories)

        if options == {"growth"}:
            try:
                self._ax = ca.category_int_growth(categories, metadata)
            except TypeError:
                self._ax = ca.category_str_growth(categories, metadata)
        elif options == set():
            try:
                self._ax = ca.category_int(categories, metadata)
            except TypeError:
                self._ax = ca.category_str(categories, metadata)
        else:
            raise KeyError("Unsupported collection of options")


Category.__module__ = "boost_histogram.axis"
