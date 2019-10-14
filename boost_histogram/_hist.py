from __future__ import absolute_import, division, print_function

from .utils import FactoryMeta, KWArgs

from . import core as _core

_histograms = (
    _core.hist._any_double,
    _core.hist._any_int,
    _core.hist._any_atomic_int,
    _core.hist._any_unlimited,
    _core.hist._any_weight,
    _core.hist._any_mean,
    _core.hist._any_weighted_mean,
)


def _arg_shortcut(item):
    if isinstance(item, tuple):
        return _core.axis._regular_uoflow(*item)
    else:
        return item


def _make_histogram(*args, **kwargs):
    """
    Make a histogram with an optional storage (keyword only).
    """

    with KWArgs(kwargs) as k:
        storage = k.optional("storage", _core.storage.double())

    # Initialize storage if user has not
    if isinstance(storage, type):
        storage = storage()

    args = [_arg_shortcut(arg) for arg in args]

    if len(args) > _core.hist._axes_limit:
        raise IndexError(
            "Too many axes, must be less than {}".format(_core.hist._axes_limit)
        )

    for h in _histograms:
        if isinstance(storage, h._storage_type):
            return h(args, storage)

    raise TypeError("Unsupported storage")


histogram = FactoryMeta(_make_histogram, _histograms)


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


def _compute_commonindex(self, index, expand):
    # Normalize -> h[i] == h[i,]
    if not isinstance(index, tuple):
        index = (index,)

    # Now a list
    if expand:
        indexes = _expand_ellipsis(index, self.rank)
    else:
        indexes = list(index)

    if len(indexes) != self.rank:
        raise IndexError("IndexError: Wrong number of indices for histogram")

    # Allow [bh.loc(...)] to work
    for i in range(len(indexes)):
        if hasattr(indexes[i], "value"):
            indexes[i] = self.axis(i).index(indexes[i].value)
        elif hasattr(indexes[i], "flow"):
            if indexes[i].flow == 1:
                indexes[i] = self.axis(i).size
            elif indexes[i].flow == -1:
                indexes[i] = -1
        elif isinstance(indexes[i], int):
            if abs(indexes[i]) >= self.axis(i).size:
                raise IndexError("histogram index is out of range")
            indexes[i] %= self.axis(i).size

    return indexes


def _compute_getitem(self, index):

    indexes = _compute_commonindex(self, index, expand=True)

    # If this is (now) all integers, return the bin contents
    try:
        return self.at(*indexes)
    except RuntimeError:
        pass

    integrations = set()
    slices = []

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
                        if ind.start is not None or ind.stop is not None:
                            raise IndexError(
                                "Currently cut projections are not supported"
                            )
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
                else (self.axis(i).index(x.value) if hasattr(x, "value") else x)
            )
            begin = process_loc(ind.start, 0)
            end = process_loc(ind.stop, len(self.axis(i)))

            slices.append(_core.algorithm.slice_and_rebin(i, begin, end, merge))

    reduced = self.reduce(*slices)
    if not integrations:
        return reduced
    else:
        projections = [i for i in range(self.rank) if i not in integrations]
        return reduced.project(*projections) if projections else self.sum(flow=True)


def _compute_setitem(self, index, value):
    indexes = _compute_commonindex(self, index, expand=True)

    self._at_set(value, *indexes)


for h in _histograms:
    h.__getitem__ = _compute_getitem
    h.__setitem__ = _compute_setitem

del h
