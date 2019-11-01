from __future__ import absolute_import, division, print_function

from .kwargs import KWArgs

from .. import _core
from .axis import _to_axis, Axis as _Axis
from .axistuple import AxesTuple

import warnings
import numpy as np

_histograms = (
    _core.hist.any_double,
    _core.hist.any_int,
    _core.hist.any_atomic_int,
    _core.hist.any_unlimited,
    _core.hist.any_weight,
    _core.hist.any_mean,
    _core.hist.any_weighted_mean,
)


def _arg_shortcut(item):
    if isinstance(item, tuple):
        return _core.axis.regular_uoflow(*item)
    elif isinstance(item, _Axis):
        return item._ax
    else:
        return item
        # TODO: This currently support raw axis object for old tests.
        # Replace with:
        # raise TypeError("Only axes supported in histogram constructor")


def _expand_ellipsis(indexes, rank):
    indexes = list(indexes)
    number_ellipses = indexes.count(Ellipsis)
    if number_ellipses == 0:
        return indexes
    elif number_ellipses == 1:
        index = indexes.index(Ellipsis)
        additional = rank + 1 - len(indexes)
        if additional < 0:
            raise IndexError("too many indices for histogram")

        # Fill out the ellipsis with empty slices
        return indexes[:index] + [slice(None)] * additional + indexes[index + 1 :]

    else:
        raise IndexError("an index can only have a single ellipsis ('...')")


def _compute_commonindex(hist, index, expand):
    # Normalize -> h[i] == h[i,]
    if not isinstance(index, tuple):
        index = (index,)

    # Now a list
    if expand:
        indexes = _expand_ellipsis(index, hist.rank())
    else:
        indexes = list(index)

    if len(indexes) != hist.rank():
        raise IndexError("Wrong number of indices for histogram")

    # Allow [bh.loc(...)] to work
    for i in range(len(indexes)):
        if callable(indexes[i]):
            indexes[i] = indexes[i](_to_axis(hist.axis(i)))
        elif hasattr(indexes[i], "flow"):
            if indexes[i].flow == 1:
                indexes[i] = hist.axis(i).size
            elif indexes[i].flow == -1:
                indexes[i] = -1
        elif isinstance(indexes[i], int):
            if abs(indexes[i]) >= hist.axis(i).size:
                raise IndexError("histogram index is out of range")
            indexes[i] %= hist.axis(i).size

    return indexes


class BaseHistogram(object):
    def __init__(self, *args, **kwargs):
        """
        Construct a new histogram.

        If you pass in a single argument, this will be treated as a
        histogram and this will convert the histogram to this type of
        histogram (DensityHistogram, Histogram, BoostHistogram).

        Parameters
        ----------
        *args : Axis
            Provide 1 or more axis instances.
        storage : Storage = bh.storage.double
            Select a storage to use in the histogram
        """

        # Allow construction from a raw histogram object (internal)
        if not kwargs and len(args) == 1 and isinstance(args[0], _histograms):
            self._hist = args[0]
            return

        if not kwargs and len(args) == 1 and isinstance(args[0], BaseHistogram):
            self._hist = args[0]._hist.__copy__()  # Replace with copy?
            return

        # Keyword only trick (change when Python2 is dropped)
        with KWArgs(kwargs) as k:
            storage = k.optional("storage", _core.storage.double())

        # Initialize storage if user has not
        if isinstance(storage, type):
            storage = storage()

        # Allow a tuple to represent a regular axis
        args = [_arg_shortcut(arg) for arg in args]

        if len(args) > _core.hist._axes_limit:
            raise IndexError(
                "Too many axes, must be less than {}".format(_core.hist._axes_limit)
            )

        # Check all available histograms, and if the storage matches, return that one
        for h in _histograms:
            if isinstance(storage, h._storage_type):
                self._hist = h(args, storage)
                return

        raise TypeError("Unsupported storage")

    def __repr__(self):
        return self.__class__.__name__ + repr(self._hist)[9:]

    def __array__(self):
        return np.asarray(self._hist)
        # TODO: .view does not seem to return an editable view
        #        so we have to use the buffer interface here

    def reset(self):
        """
        Reset bin counters to default values.
        """
        self._hist.reset()
        return self

    def empty(self, flow=False):
        """
        Check to see if the histogram has any non-default values.
        You can use flow=True to check flow bins too.
        """
        return self._hist.empty(flow)

    def __add__(self, other):
        return self.__class__(self._hist + other._hist)

    def __eq__(self, other):
        return self._hist == other._hist

    def __ne__(self, other):
        return self._hist != other._hist

    # If these fail, the underlying object throws the correct error
    def __mul__(self, other):
        if isinstance(other, BaseHistogram):
            return self.__class__(self._hist * other._hist)
        else:
            return self.__class__(self._hist * other)

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        return self.__class__(self._hist.__imul__(other))

    def __truediv__(self, other):
        return self.__class__(self._hist.__truediv__(other._hist))

    def __div__(self, other):
        return self.__class__(self._hist.__div__(other._hist))

    def __itruediv__(self, other):
        return self.__class__(self._hist.__itruediv__(other._hist))

    def __idiv__(self, other):
        return self.__class__(self._hist.__idiv__(other._hist))

    def reduce(self, *args):
        """
        Reduce based on one or more reduce_option. Generally,
        the [] indexing is easier, but this might be useful in
        some cases, and is lighter-weight.
        """

        return self.__class__(self._hist.reduce(*args))

    def project(self, *args):
        """
        Project to a single axis or several axes on a multidiminsional histogram. Provided a list of axis numbers, this will produce the histogram over those axes only. Flow bins are used if available.
        """

        return self.__class__(self._hist.project(*args))

    def fill(self, *args, **kwargs):
        """
        Insert data into the histogram.

        Parameters
        ----------
        *args : Union[Array[float], Array[int], Array[str], float, int, str]
            Provide one value or array per dimension.
        weight : List[Union[Array[float], Array[int], Array[str], float, int, str]]]
            Provide weights (only if the histogram storage supports it)
        sample : List[Union[Array[float], Array[int], Array[str], float, int, str]]]
            Provide samples (only if the histogram storage supports it)

        """

        self._hist.fill(*args, **kwargs)
        return self

    def sum(self, flow=False):
        """
        Compute the sum over the histogram bins (optionally including the flow bins).
        """
        return self._hist.sum(flow)

    def _axis(self, i):
        """
        Get N-th axis.
        """
        return _to_axis(self._hist.axis(i))


class BoostHistogram(BaseHistogram):
    axis = BaseHistogram._axis

    def rank(self):
        """
        Number of axes (dimensions) of histogram.
        """
        return self._hist.rank()

    def size(self):
        """
        Total number of bins in the histogram (including underflow/overflow).
        """
        return self._hist.size()

    def at(self, *indexes):
        """
        Select a contents given indices. -1 is the underflow bin, N is the overflow bin.
        """
        return self._hist.at(*indexes)

    # Call uses fill since it supports strings,
    # runtime argument list, etc.
    def __call__(self, *args, **kargs):
        args = (((a,) if isinstance(a, str) else a) for a in args)
        self._hist.fill(*args, **kargs)
        return self


class Histogram(BaseHistogram):
    def __init__(self, *args, **kwargs):
        super(Histogram, self).__init__(*args, **kwargs)

        # If this is a property, tab completion in IPython does not work
        self.axes = AxesTuple(self._axis(i) for i in range(self.rank))

    __init__.__doc__ = BaseHistogram.__doc__

    def to_numpy(self, flow=False):
        """
        Convert to a Numpy style tuple of return arrays.

        Return
        ------
        contents : Array[Any]
            The bin contents
        *edges : Array[float]
            The edges for each dimension
        """
        return self._hist.to_numpy(flow)

    def view(self, flow=False):
        """
        Return a view into the data, optionally with overflow turned on.
        """
        return self._hist.view(flow)

    def _at(self, *ind):
        "Temporary workaround for looping from -1 in old tests"
        return self._hist.at(*ind)

    @property
    def rank(self):
        """
        Number of axes (dimensions) of histogram.
        """
        return self._hist.rank()

    @property
    def size(self):
        """
        Total number of bins in the histogram (including underflow/overflow).
        """
        return self._hist.size()

    def __getitem__(self, index):

        indexes = _compute_commonindex(self._hist, index, expand=True)

        # If this is (now) all integers, return the bin contents
        try:
            return self._hist.at(*indexes)
        except RuntimeError:
            pass

        integrations = set()
        slices = []
        zeroes_start = []
        zeroes_stop = []

        # Compute needed slices and projections
        for i, ind in enumerate(indexes):
            if not isinstance(ind, slice):
                raise IndexError(
                    "Invalid arguments as an index, use all integers "
                    "or all slices, and do not mix"
                )
            if ind != slice(None):
                merge = 1
                if ind.step is not None:
                    if hasattr(ind.step, "projection"):
                        if ind.step.projection:
                            integrations.add(i)
                            if ind.start is not None:  # TODO: Support callables too
                                zeroes_start.append(i)
                            if ind.stop is not None:
                                zeroes_stop.append(i)
                        elif hasattr(ind.step, "factor"):
                            merge = ind.step.factor
                        else:
                            raise IndexError("Invalid rebin, must have integer .factor")
                    else:
                        raise IndexError(
                            "The third argument to a slice must be rebin or projection"
                        )

                process_loc = (
                    lambda x, y: y
                    if x is None
                    else x(self._axis(i))
                    if callable(x)
                    else x
                )
                begin = process_loc(ind.start, 0)
                end = process_loc(ind.stop, len(self._axis(i)))

                slices.append(_core.algorithm.slice_and_rebin(i, begin, end, merge))

        reduced = self.reduce(*slices)
        if not integrations:
            return self.__class__(reduced)
        else:
            projections = [i for i in range(self.rank) if i not in integrations]

            # Replacement for crop missing in BH
            for i in zeroes_start:
                if self.axes[i].options.underflow:
                    reduced._hist._reset_row(i, -1)
            for i in zeroes_stop:
                if self.axes[i].options.underflow:
                    reduced._hist._reset_row(i, reduced.axes[i].size)

            return (
                self.__class__(reduced.project(*projections))
                if projections
                else reduced.sum(flow=True)
            )

    def __setitem__(self, index, value):
        indexes = _compute_commonindex(self._hist, index, expand=True)
        self._hist._at_set(value, *indexes)
