from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from .._core.axis import options

from .._core import axis as ca

from .kwargs import KWArgs
from .sig_tools import inject_signature
from .axis_transform import AxisTransform


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
        return repr(self._ax)

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


class Regular(Axis):
    __slots__ = ()
    _CLASSES = {
        ca.regular_uoflow,
        ca.regular_uoflow_growth,
        ca.regular_uflow,
        ca.regular_oflow,
        ca.regular_none,
        ca.regular_numpy,
        ca.regular_sqrt,
        ca.regular_pow,
        ca.regular_log,
        ca.circular,
    }

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
        metadata : object
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


class Variable(Axis):
    __slots__ = ()
    _CLASSES = {
        ca.variable_none,
        ca.variable_uflow,
        ca.variable_oflow,
        ca.variable_uoflow,
        ca.variable_uoflow_growth,
    }

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


class Integer(Axis):
    __slots__ = ()
    _CLASSES = {
        ca.integer_none,
        ca.integer_uflow,
        ca.integer_oflow,
        ca.integer_uoflow,
        ca.integer_growth,
    }

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


class Category(Axis):
    __slots__ = ()
    _CLASSES = {
        ca.category_int_growth,
        ca.category_str_growth,
        ca.category_int,
        ca.category_str,
    }

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


def _walk_subclasses(cls):
    for base in cls.__subclasses__():
        yield base
        for inner in _walk_subclasses(base):
            yield inner


def _to_axis(ax):
    for base in _walk_subclasses(Axis):
        if ax.__class__ in base._CLASSES:
            nice_ax = base.__new__(base)
            nice_ax._ax = ax
            return nice_ax

    raise TypeError("Invalid axes passed in")
