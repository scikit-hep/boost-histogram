# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import copy
import sys
import threading
import warnings
from typing import Any, Optional, Tuple

import numpy as np

from .. import _core
from .axestuple import AxesTuple
from .axis import Axis
from .enum import Kind
from .kwargs import KWArgs
from .sig_tools import inject_signature
from .six import string_types
from .storage import Double, Storage
from .utils import MAIN_FAMILY, cast, register, set_family, set_module
from .view import _to_view

if sys.version_info >= (3, 4):
    from os import cpu_count
else:
    from multiprocessing import cpu_count

ArrayLike = Any


NOTHING = object()

_histograms = (
    _core.hist.any_double,
    _core.hist.any_int64,
    _core.hist.any_atomic_int64,
    _core.hist.any_unlimited,
    _core.hist.any_weight,
    _core.hist.any_mean,
    _core.hist.any_weighted_mean,
)


def _fill_cast(value, inner=False):
    """
    Convert to NumPy arrays. Some buffer objects do not get converted by forcecast.
    If not called by itself (inner=False), then will work through one level of tuple/list.
    """
    if value is None or isinstance(value, string_types + (bytes,)):
        return value
    elif not inner and isinstance(value, (tuple, list)):
        return tuple(_fill_cast(a, inner=True) for a in value)
    elif hasattr(value, "__iter__") or hasattr(value, "__array__"):
        return np.asarray(value)
    else:
        return value


def _arg_shortcut(item):
    msg = "Developer shortcut: will be removed in a future version"
    if isinstance(item, tuple) and len(item) == 3:
        warnings.warn(msg, FutureWarning)
        return _core.axis.regular_uoflow(item[0], item[1], item[2])
    elif isinstance(item, Axis):
        return item._ax
    else:
        raise TypeError("Only axes supported in histogram constructor")


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


# We currently do not cast *to* a histogram, but this is consistent
# and could be used later.
@register(_histograms)
@set_family(MAIN_FAMILY)
@set_module("boost_histogram")
class Histogram(object):
    # Note this is a __slots__ __dict__ class!
    __slots__ = (
        "_hist",
        "axes",
        "__dict__",
    )
    # .metadata and ._variance_known are part of the dict

    @inject_signature(
        "self, *axes, storage=Double(), metadata=None", locals={"Double": Double}
    )
    def __init__(self, *axes, **kwargs):
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

        # Allow construction from a raw histogram object (internal)
        if len(axes) == 1 and isinstance(axes[0], _histograms):
            self._hist = axes[0]
            self.metadata = kwargs.get("metadata")
            self.axes = self._generate_axes_()
            return

        # If we construct with another Histogram as the only positional argument,
        # support that too
        if len(axes) == 1 and isinstance(axes[0], Histogram):
            # Special case - we can recursively call __init__ here
            self.__init__(axes[0]._hist)  # type: ignore
            self._from_histogram_object(axes[0])
            return

        # Support objects that provide a to_boost method, like Uproot
        elif len(axes) == 1 and hasattr(axes[0], "_to_boost_histogram_"):
            self.__init__(axes[0]. _to_boost_histogram_())
            return

        # Keyword only trick (change when Python2 is dropped)
        with KWArgs(kwargs) as k:
            storage = k.optional("storage")
            if storage is None:
                storage = Double()
            self.metadata = k.optional("metadata")

        # Check for missed parenthesis or incorrect types
        if not isinstance(storage, Storage):
            if issubclass(storage, Storage):
                raise KeyError(
                    "Passing in an initialized storage has been removed. Please add ()."
                )
            else:
                raise KeyError("Only storages allowed in storage argument")

        # Allow a tuple to represent a regular axis
        axes = tuple(_arg_shortcut(arg) for arg in axes)

        if len(axes) > _core.hist._axes_limit:
            raise IndexError(
                "Too many axes, must be less than {}".format(_core.hist._axes_limit)
            )

        # Check all available histograms, and if the storage matches, return that one
        for h in _histograms:
            if isinstance(storage, h._storage_type):
                self._hist = h(axes, storage)
                self.axes = self._generate_axes_()
                return

        raise TypeError("Unsupported storage")

    def _from_histogram_object(self, other):
        """
        Return a new histogram object, possibly converting from a different subclass.
        """
        self._hist = other._hist
        self.__dict__ = copy.copy(other.__dict__)
        self.axes = self._generate_axes_()
        for ax in self.axes:
            ax.__dict__ = copy.copy(ax._ax.metadata)

        # Allow custom behavior on either "from" or "to"
        other._export_bh_(self)
        self._import_bh_()

    def _import_bh_(self):
        """
        If any post-processing is needed to pass a histogram between libraries, a
        subclass can implement it here. self is the new instance in the current
        (converted-to) class.
        """

    @classmethod
    def _export_bh_(cls, self):
        """
        If any preparation is needed to pass a histogram between libraries, a subclass can
        implement it here. cls is the current class being converted from, and self is the
        instance in the class being converted to.
        """

    def _generate_axes_(self):
        """
        This is called to fill in the axes. Subclasses can override it if they need
        to change the axes tuple.
        """

        return AxesTuple(self._axis(i) for i in range(self.ndim))

    def _new_hist(self, _hist, memo=NOTHING):
        """
        Return a new histogram given a new _hist, copying metadata.
        """

        other = self.__class__(_hist)
        if memo is NOTHING:
            other.__dict__ = copy.copy(self.__dict__)
        else:
            other.__dict__ = copy.deepcopy(self.__dict__, memo)
        other.axes = other._generate_axes_()

        for ax in other.axes:
            if memo is NOTHING:
                ax.__dict__ = copy.copy(ax._ax.metadata)
            else:
                ax.__dict__ = copy.deepcopy(ax._ax.metadata, memo)

        return other

    @property
    def ndim(self):
        """
        Number of axes (dimensions) of the histogram.
        """
        return self._hist.rank()

    def view(self, flow=False):
        """
        Return a view into the data, optionally with overflow turned on.
        """
        return _to_view(self._hist.view(flow))

    def __array__(self):
        return self.view(False)

    def __add__(self, other):
        result = self.copy(deep=False)
        return result.__iadd__(other)

    def __iadd__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            return self
        self._compute_inplace_op("__iadd__", other)

        # Addition may change the axes if they can grow
        self.axes = self._generate_axes_()

        return self

    def __radd__(self, other):
        return self + other

    def __eq__(self, other):
        return self._hist == other._hist

    def __ne__(self, other):
        return self._hist != other._hist

    # If these fail, the underlying object throws the correct error
    def __mul__(self, other):
        result = self.copy(deep=False)
        return result._compute_inplace_op("__imul__", other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        result = self.copy(deep=False)
        return result._compute_inplace_op("__itruediv__", other)

    def __div__(self, other):
        result = self.copy(deep=False)
        return result._compute_inplace_op("__idiv__", other)

    def _compute_inplace_op(self, name, other):
        if isinstance(other, Histogram):
            getattr(self._hist, name)(other._hist)
        elif isinstance(other, _histograms):
            getattr(self._hist, name)(other)
        elif hasattr(other, "shape") and other.shape:
            if len(other.shape) != self.ndim:
                raise ValueError(
                    "Number of dimensions {} must match histogram {}".format(
                        len(other.shape), self.ndim
                    )
                )
            elif all((a == b or a == 1) for a, b in zip(other.shape, self.shape)):
                view = self.view(flow=False)
                getattr(view, name)(other)
            elif all((a == b or a == 1) for a, b in zip(other.shape, self.axes.extent)):
                view = self.view(flow=True)
                getattr(view, name)(other)
            else:
                raise ValueError(
                    "Wrong shape {}, expected {} or {}".format(
                        other.shape, self.shape, self.axes.extent
                    )
                )
        else:
            view = self.view(flow=False)
            getattr(view, name)(other)
        self._variance_known = False
        return self

    def __idiv__(self, other):
        return self._compute_inplace_op("__idiv__", other)

    def __itruediv__(self, other):
        return self._compute_inplace_op("__itruediv__", other)

    def __imul__(self, other):
        return self._compute_inplace_op("__imul__", other)

    # TODO: Marked as too complex by flake8. Should be factored out a bit.
    @inject_signature("self, *args, weight=None, sample=None, threads=None")
    def fill(self, *args, **kwargs):  # noqa: C901
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
        threads : Optional[int]
            Fill with threads. Defaults to None, which does not activate
            threaded filling.  Using 0 will automatically pick the number of
            available threads (usually two per core).
        """

        with KWArgs(kwargs) as kw:
            weight = kw.optional("weight")
            sample = kw.optional("sample")
            threads = kw.optional("threads")

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
        args = _fill_cast(args)
        weight = _fill_cast(weight)
        sample = _fill_cast(sample)

        if threads is None or threads == 1:
            self._hist.fill(*args, weight=weight, sample=sample)
            return self

        if threads == 0:
            threads = cpu_count()

        if self._hist._storage_type in {
            _core.storage.mean,
            _core.storage.weighted_mean,
        }:
            raise RuntimeError("Mean histograms do not support threaded filling")

        data = [np.array_split(a, threads) for a in args]

        if weight is None or np.isscalar(weight):
            weights = [weight] * threads
        else:
            weights = np.array_split(weight, threads)

        if sample is None or np.isscalar(sample):
            samples = [sample] * threads
        else:
            samples = np.array_split(sample, threads)

        if self._hist._storage_type is _core.storage.atomic_int64:

            def fun(weight, sample, *args):
                self._hist.fill(*args, weight=weight, sample=sample)

        else:
            sum_lock = threading.Lock()

            def fun(weight, sample, *args):
                local_hist = self._hist.__copy__()
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

    def __str__(self):
        """
        A rendering of the histogram is made using ASCII or unicode characters
        (whatever is supported by the terminal). What exactly is displayed is
        still experimental. Do not rely on any particular rendering.
        """
        # TODO check the terminal width and adjust the presentation
        # only use for 1D, fall back to repr for ND
        if self._hist.rank() == 1:
            s = str(self._hist)
            # get rid of first line and last character
            s = s[s.index("\n") + 1 : -1]
        else:
            s = repr(self)
        return s

    def _axis(self, i=0):
        """
        Get N-th axis.
        """
        return cast(self, self._hist.axis(i), Axis)

    @property
    def _storage_type(self):
        return cast(self, self._hist._storage_type, Storage)

    def _reduce(self, *args):
        return self._new_hist(self._hist.reduce(*args))

    def __copy__(self):
        return self._new_hist(copy.copy(self._hist))

    def __deepcopy__(self, memo):
        return self._new_hist(copy.deepcopy(self._hist), memo=memo)

    def __getstate__(self):
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

    def __setstate__(self, state):
        if isinstance(state, tuple):
            if state[0] == 0:
                for key, value in state[1].items():
                    setattr(self, key, value)

                # Added in 0.12
                if "_variance_known" not in state[1]:
                    self._variance_known = True
            else:
                msg = "Cannot open boost-histogram pickle v{}".format(state[0])
                raise RuntimeError(msg)

            self.axes = self._generate_axes_()

        else:  # Classic (0.10 and before) state
            self._hist = state["_hist"]
            self._variance_known = True
            self.metadata = state.get("metadata", None)
            for i in range(self._hist.rank()):
                self._hist.axis(i).metadata = {"metadata": self._hist.axis(i).metadata}
            self.axes = self._generate_axes_()

    def __repr__(self):
        newline = "\n  "
        sep = "," if len(self.axes) > 0 else ""
        ret = "{self.__class__.__name__}({newline}".format(
            self=self, newline=newline if len(self.axes) > 1 else ""
        )
        ret += ",{newline}".format(newline=newline).join(repr(ax) for ax in self.axes)
        ret += "{comma}{newline}storage={storage}".format(
            storage=self._storage_type(),
            newline=newline
            if len(self.axes) > 1
            else " "
            if len(self.axes) > 0
            else "",
            comma=sep,
        )
        ret += ")"
        outer = self.sum(flow=True)
        if outer:
            inner = self.sum(flow=False)
            ret += " # Sum: {}".format(inner)
            if inner != outer:
                ret += " ({} with flow)".format(outer)
        return ret

    def _compute_commonindex(self, index):
        """
        Takes indices and returns two iterables; one is a tuple or dict of the
        original, Ellipsis expanded index, and the other returns index,
        operation value pairs.
        """
        # Shorten the computations with direct access to raw object
        hist = self._hist

        # Support dict access
        if hasattr(index, "items"):
            indexes = [slice(None)] * hist.rank()
            for k, v in index.items():
                indexes[k] = v

        # Normalize -> h[i] == h[i,]
        else:
            if not isinstance(index, tuple):
                index = (index,)
            # Now a list
            indexes = _expand_ellipsis(index, hist.rank())

        if len(indexes) != hist.rank():
            raise IndexError("Wrong number of indices for histogram")

        # Allow [bh.loc(...)] to work
        for i in range(len(indexes)):
            # Support sum and rebin directly
            if indexes[i] is sum or hasattr(indexes[i], "factor"):
                indexes[i] = slice(None, None, indexes[i])
            # General locators
            # Note that MyPy doesn't like these very much - the fix
            # will be to properly set input types
            elif callable(indexes[i]):
                indexes[i] = indexes[i](self.axes[i])  # type: ignore
            elif hasattr(indexes[i], "__index__"):
                if abs(indexes[i]) >= hist.axis(i).size:  # type: ignore
                    raise IndexError("histogram index is out of range")
                indexes[i] %= hist.axis(i).size

        return indexes

    @inject_signature("self, flow=False, *, dd=False, view=False")
    def to_numpy(self, flow=False, **kwargs):
        """
        Convert to a Numpy style tuple of return arrays. Edges are converted to
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

        with KWArgs(kwargs) as kw:
            dd = kw.optional("dd", False)
            view = kw.optional("view", False)

        # Python 3+ would be simpler
        return_tuple = self._hist.to_numpy(flow)
        hist = return_tuple[0]

        if view:
            hist = self.view(flow=flow)
        else:
            hist = self.values(flow=flow)

        if dd:
            return hist, return_tuple[1:]
        else:
            return (hist,) +  return_tuple[1:]

    @inject_signature("self, *, deep=True")
    def copy(self, **kwargs):
        """
        Make a copy of the histogram. Defaults to making a
        deep copy (axis metadata copied); use deep=False
        to avoid making a copy of axis metadata.
        """

        # Future versions may add new options here
        with KWArgs(kwargs) as k:
            deep = k.optional("deep", True)

        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def reset(self):
        """
        Reset bin counters to default values.
        """
        self._hist.reset()
        return self

    def empty(self, flow=False):
        # type: (bool) -> bool
        """
        Check to see if the histogram has any non-default values.
        You can use flow=True to check flow bins too.
        """
        return self._hist.empty(flow)

    def sum(self, flow=False):
        """
        Compute the sum over the histogram bins (optionally including the flow bins).
        """
        return self._hist.sum(flow)

    @property
    def size(self):
        # type: () -> int
        """
        Total number of bins in the histogram (including underflow/overflow).
        """
        return self._hist.size()

    @property
    def shape(self):
        # type: () -> Tuple[int, ...]
        """
        Tuple of axis sizes (not including underflow/overflow).
        """
        return self.axes.size

    # TODO: Marked as too complex by flake8. Should be factored out a bit.
    def __getitem__(self, index):  # noqa: C901

        indexes = self._compute_commonindex(index)

        # If this is (now) all integers, return the bin contents
        # But don't try *dict!
        if not hasattr(indexes, "items"):
            try:
                return self._hist.at(*indexes)
            except RuntimeError:
                pass

        integrations = set()
        slices = []

        # Compute needed slices and projections
        for i, ind in enumerate(indexes):
            if hasattr(ind, "__index__"):
                ind = slice(ind.__index__(), ind.__index__() + 1, sum)

            elif not isinstance(ind, slice):
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

            if ind != slice(None):
                merge = 1
                if ind.step is not None:
                    if hasattr(ind.step, "factor"):
                        merge = ind.step.factor
                    elif callable(ind.step):
                        if ind.step is sum:
                            integrations.add(i)
                        else:
                            raise RuntimeError("Full UHI not supported yet")

                        if ind.start is not None or ind.stop is not None:
                            slices.append(
                                _core.algorithm.slice(
                                    i, start, stop, _core.algorithm.slice_mode.crop
                                )
                            )
                        continue
                    else:
                        raise IndexError(
                            "The third argument to a slice must be rebin or projection"
                        )

                slices.append(_core.algorithm.slice_and_rebin(i, start, stop, merge))

        reduced = self._hist.reduce(*slices)

        if not integrations:
            return self._new_hist(reduced)
        else:
            projections = [i for i in range(self.ndim) if i not in integrations]

            return (
                self._new_hist(reduced.project(*projections))
                if projections
                else reduced.sum(flow=True)
            )

    def __setitem__(self, index, value):
        """
        There are several supported possibilities:

            h[slice] = array # same size

        If an array is given to a compatible slice, it is set.

            h[a:] = array # One larger

        If an array is given that does not match, if it does match the
        with-overflow size, it fills that.

        PLANNED (not yet supported):

            h[a:] = h2

        If another histogram is given, that must either match with or without
        overflow, where the overflow bins must be overflow bins (that is,
        you cannot set a histogram's flow bins from another histogram that
        is 2 larger). Bin edges must be a close match, as well. If you don't
        want this level of type safety, just use ``h[...] = h2.view()``.
        """
        indexes = self._compute_commonindex(index)

        if isinstance(value, Histogram):
            raise TypeError("Not supported yet")

        value = np.asarray(value)
        view = self.view(flow=True)

        # Support raw arrays for accumulators, the final dimension is the constructor values
        if (
            value.ndim > 0
            and len(view.dtype) > 0
            and len(value.dtype) == 0
            and len(view.dtype) == value.shape[-1]
        ):
            value_shape = value.shape[:-1]
            value_ndim = value.ndim - 1
        else:
            value_shape = value.shape
            value_ndim = value.ndim

        # Numpy does not broadcast partial slices, but we would need
        # to allow it (because we do allow broadcasting up dimensions)
        # Instead, we simply require matching dimensions.
        if value_ndim > 0 and value_ndim != sum(isinstance(i, slice) for i in indexes):
            raise ValueError(
                "Setting a {}D histogram with a {}D array must have a matching number of dimensions".format(
                    len(indexes), value_ndim
                )
            )

        # Here, value_n does not increment with n if this is not a slice
        value_n = 0
        for n, request in enumerate(indexes):
            has_underflow = self.axes[n].traits.underflow
            has_overflow = self.axes[n].traits.overflow

            if isinstance(request, slice):
                # Only consider underflow/overflow if the endpoints are not given
                use_underflow = has_underflow and request.start is None
                use_overflow = has_overflow and request.stop is None

                # Make the limits explicit since we may need to shift them
                start = 0 if request.start is None else request.start
                stop = len(self.axes[n]) if request.stop is None else request.stop
                request_len = stop - start

                # If set to a scalar, then treat it like broadcasting without flow bins
                if value_ndim == 0:
                    start = 0 + has_overflow
                    stop = len(self.axes[n]) + has_underflow

                # Normal setting
                elif request_len == value_shape[value_n]:
                    start += has_underflow
                    stop += has_underflow

                # Expanded setting
                elif request_len + use_underflow + use_overflow == value_shape[value_n]:
                    start += has_underflow and not use_underflow
                    stop += has_underflow + (has_overflow and use_overflow)

                # Single element broadcasting
                elif value_shape[value_n] == 1:
                    start += has_underflow
                    stop += has_underflow

                else:
                    msg = "Mismatched shapes in dimension {}".format(n)
                    msg += ", {} != {}".format(value_shape[n], request_len)
                    if use_underflow or use_overflow:
                        msg += " or {}".format(
                            request_len + use_underflow + use_overflow
                        )
                    raise ValueError(msg)
                indexes[n] = slice(start, stop, request.step)
                value_n += 1
            else:
                indexes[n] = request + has_underflow

        view[tuple(indexes)] = value

    def project(self, *args):
        # type: (Axis) -> Histogram
        """
        Project to a single axis or several axes on a multidimensional histogram.
        Provided a list of axis numbers, this will produce the histogram over
        those axes only. Flow bins are used if available.
        """

        return self._new_hist(self._hist.project(*args))

    # Implementation of PlottableHistogram

    @property
    def kind(self):
        # type: () -> Kind
        """
        Returns Kind.COUNT if this is a normal summing histogram, and Kind.MEAN if this is a
        mean histogram.

        :return: Kind
        """
        if self._hist._storage_type in {
            _core.storage.mean,
            _core.storage.weighted_mean,
        }:
            return Kind.MEAN
        else:
            return Kind.COUNT

    def values(self, flow=False):
        # type: (bool) -> ArrayLike
        """
        Returns the accumulated values. The counts for simple histograms, the
        sum of weights for weighted histograms, the mean for profiles, etc.

        If counts is equal to 0, the value in that cell is undefined if
        kind == "MEAN".

        :param flow: Enable flow bins. Not part of PlottableHistogram, but
        included for consistency with other methods and flexibility.

        :return: np.ndarray[np.float64]
        """

        view = self.view(flow)
        if len(view.dtype) == 0:
            return view
        else:
            return view.value

    def variances(self, flow=False):
        # type: (bool) -> Optional[ArrayLike]
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

        :return: np.ndarray[np.float64]
        """

        view = self.view(flow)
        if len(view.dtype) == 0:
            if self._variance_known:
                return view
            else:
                return None
        elif hasattr(view, "sum_of_weights"):
            return np.divide(
                view.variance,
                view.sum_of_weights,
                out=np.full(view.sum_of_weights.shape, np.nan),
                where=view.sum_of_weights > 1,
            )

        elif hasattr(view, "count"):
            return np.divide(
                view.variance,
                view.count,
                out=np.full(view.count.shape, np.nan),
                where=view.count > 1,
            )
        else:
            return view.variance

    def counts(self, flow=False):
        # type: (bool) -> Optional[ArrayLike]
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

        :return: np.ndarray[np.float64]
        """

        view = self.view(flow)

        if len(view.dtype) == 0:
            return view
        elif hasattr(view, "sum_of_weights"):
            return np.divide(
                view.sum_of_weights ** 2,
                view.sum_of_weights_squared,
                out=np.zeros_like(view.sum_of_weights, dtype=np.float64),
                where=view.sum_of_weights_squared != 0,
            )
        elif hasattr(view, "count"):
            return view.count
        else:
            return view.value
