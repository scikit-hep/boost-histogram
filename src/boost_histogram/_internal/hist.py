# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import copy
import os
import threading
import warnings

import numpy as np

from .. import _core
from .axis import Axis
from .axestuple import AxesTuple
from .kwargs import KWArgs
from .sig_tools import inject_signature
from .six import string_types
from .storage import Double, Storage
from .utils import cast, register, set_family, MAIN_FAMILY, set_module
from .view import _to_view


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
        # TODO: Currently segfaults if we pass in a non-axis to the C++ code
        # Using the public interface above, this should never be possible.


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

        # Allow construction from a raw histogram object (internal)
        if len(axes) == 1 and isinstance(axes[0], _histograms):
            self._hist = axes[0]
            self.metadata = kwargs.get("metadata")
            self.axes = self._generate_axes_()
            return

        # If we construct with another Histogram as the only positional argument,
        # support that too
        if len(axes) == 1 and isinstance(axes[0], Histogram):
            self.__init__(axes[0]._hist)
            self._from_histogram_object(axes[0])
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

    def _from_histogram_object(self, h):
        self.__dict__ = copy.copy(h.__dict__)
        self.axes = self._generate_axes_()
        for ax in self.axes:
            ax._ax.metadata = copy.copy(ax._ax.metadata)

        # Allow custom behavior on either "from" or "to"
        h._export_bh_(self)
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
        for item in self.__dict__:
            if item not in ["axes", "_hist"]:
                if memo is NOTHING:
                    other.__dict__[item] = self.__dict__[item]
                else:
                    other.__dict__[item] = copy.deepcopy(self.__dict__[item], memo)
        other.axes = other._generate_axes_()
        for ax in other.axes:
            if memo is NOTHING:
                ax._ax.metadata = copy.copy(ax._ax.metadata)
            else:
                ax._ax.metadata = copy.deepcopy(ax._ax.metadata, memo)
        return other

    @property
    def ndim(self):
        """
        Number of axes (dimensions) of histogram.
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
                    "Number of dimensions {0} must match histogram {1}".format(
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
                    "Wrong shape {0}, expected {1} or {2}".format(
                        other.shape, self.shape, self.axes.extent
                    )
                )
        else:
            view = self.view(flow=False)
            getattr(view, name)(other)
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

        # Convert to NumPy arrays
        args = _fill_cast(args)
        weight = _fill_cast(weight)
        sample = _fill_cast(sample)

        if threads is None or threads == 1:
            self._hist.fill(*args, weight=weight, sample=sample)
            return self

        if threads == 0:
            threads = os.cpu_count()

        if (
            self._hist._storage_type is _core.storage.mean
            or self._hist._storage_type is _core.storage.weighted_mean
        ):
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

        ``dict`` contains __dict__ without "axes" and "_hist"
        """
        local_dict = copy.copy(self.__dict__)
        del local_dict["axes"]
        # Version 0 of boost-histogram pickle state
        return (0, local_dict)

    def __setstate__(self, state):
        if isinstance(state, tuple):
            if state[0] == 0:
                for key, value in state[1].items():
                    self.__dict__[key] = value
            else:
                msg = "Cannot open boost-histogram pickle v{}".format(state[0])
                raise RuntimeError(msg)

            self.axes = self._generate_axes_()

        else:  # Classic (0.10 and before) state
            self._hist = state["_hist"]
            self.metadata = state.get("metadata", None)
            self.axes = self._generate_axes_()
            for ax in self.axes:
                ax._ax.metadata = {"metadata": ax._ax.metadata}

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
            ret += " # Sum: {0}".format(inner)
            if inner != outer:
                ret += " ({0} with flow)".format(outer)
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
            elif callable(indexes[i]):
                indexes[i] = indexes[i](self.axes[i])
            elif hasattr(indexes[i], "__index__"):
                if abs(indexes[i]) >= hist.axis(i).size:
                    raise IndexError("histogram index is out of range")
                indexes[i] %= hist.axis(i).size

        return indexes

    @inject_signature("self, flow=False, *, dd=False")
    def to_numpy(self, flow=False, **kwargs):
        """
        Convert to a Numpy style tuple of return arrays.

        Parameters
        ----------

        flow : bool = False
            Include the flow bins.
        dd : bool = False
            Use the histogramdd return syntax, where the edges are in a tuple

        Return
        ------
        contents : Array[Any]
            The bin contents
        *edges : Array[float]
            The edges for each dimension
        """

        with KWArgs(kwargs) as kw:
            dd = kw.optional("dd", False)

        return_tuple = self._hist.to_numpy(flow)

        if dd:
            return return_tuple[0], return_tuple[1:]
        else:
            return return_tuple

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
    def rank(self):
        """
        Number of axes (dimensions) of histogram. DEPRECATED, use ndim.
        """
        msg = "Use .ndim instead"
        warnings.warn(msg, FutureWarning)
        return self._hist.rank()

    @property
    def size(self):
        """
        Total number of bins in the histogram (including underflow/overflow).
        """
        return self._hist.size()

    @property
    def shape(self):
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

        # Shortcut: allow raw arrays for WeightedSum
        weighted_dtype = np.dtype([("value", "<f8"), ("variance", "<f8")])
        if (
            view.dtype == weighted_dtype != value.dtype
            and len(value.dtype) != 2
            and value.ndim > 0
            and value.shape[-1] == 2
        ):
            value = value.astype(np.double).view(weighted_dtype)[..., 0]
        # Disallow mismatched data types
        elif len(value.dtype) != len(view.dtype):
            raise ValueError("Mismatched data types; matching types required")

        # Numpy does not broadcast partial slices, but we would need
        # to allow it (because we do allow broadcasting up dimensions)
        # Instead, we simply require matching dimensions.
        if value.ndim > 0 and value.ndim != sum(isinstance(i, slice) for i in indexes):
            raise ValueError(
                "Setting a {0}D histogram with a {1}D array must have a matching number of dimensions".format(
                    len(indexes), value.ndim
                )
            )

        # Here, value_n does not increment with n if this is not a slice
        value_n = 0
        for n, request in enumerate(indexes):
            has_underflow = self.axes[n].options.underflow
            has_overflow = self.axes[n].options.overflow

            if isinstance(request, slice):
                # Only consider underflow/overflow if the endpoints are not given
                use_underflow = has_underflow and request.start is None
                use_overflow = has_overflow and request.stop is None

                # Make the limits explicit since we may need to shift them
                start = 0 if request.start is None else request.start
                stop = len(self.axes[n]) if request.stop is None else request.stop
                request_len = stop - start

                # If set to a scalar, then treat it like broadcasting without flow bins
                if value.ndim == 0:
                    start = 0 + has_overflow
                    stop = len(self.axes[n]) + has_underflow

                # Normal setting
                elif request_len == value.shape[value_n]:
                    start += has_underflow
                    stop += has_underflow

                # Expanded setting
                elif request_len + use_underflow + use_overflow == value.shape[value_n]:
                    start += has_underflow and not use_underflow
                    stop += has_underflow + (has_overflow and use_overflow)

                # Single element broadcasting
                elif value.shape[value_n] == 1:
                    start += has_underflow
                    stop += has_underflow

                else:
                    msg = "Mismatched shapes in dimension {0}".format(n)
                    msg += ", {0} != {1}".format(value.shape[n], request_len)
                    if use_underflow or use_overflow:
                        msg += " or {0}".format(
                            request_len + use_underflow + use_overflow
                        )
                    raise ValueError(msg)
                indexes[n] = slice(start, stop, request.step)
                value_n += 1
            else:
                indexes[n] = request + has_underflow

        view[tuple(indexes)] = value

    def project(self, *args):
        """
        Project to a single axis or several axes on a multidiminsional histogram.
        Provided a list of axis numbers, this will produce the histogram over
        those axes only. Flow bins are used if available.
        """

        return self._new_hist(self._hist.project(*args))
