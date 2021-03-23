from functools import reduce as _reduce
from operator import mul as _mul
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple, Type, Union

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
else:
    ArrayLike = object

import numpy as _np

import boost_histogram._core as _core

from . import axis as _axis
from . import storage as _storage
from ._internal import hist as _hist
from ._internal.utils import cast as _cast

__all__ = ("histogram", "histogram2d", "histogramdd")


def histogramdd(
    a: Tuple[ArrayLike, ...],
    bins: Union[int, Tuple[int, ...]] = 10,
    range: Optional[Sequence[Union[None, Tuple[float, float]]]] = None,
    normed: None = None,
    weights: Optional[ArrayLike] = None,
    density: bool = False,
    *,
    histogram: Optional[Type[_hist.Histogram]] = None,
    storage: _storage.Storage = _storage.Double(),  # noqa: B008
    threads: Optional[int] = None
) -> Any:
    np = _np  # Hidden to keep module clean

    # TODO: Might be a bug in MyPy? This should type
    cls: Type[_hist.Histogram] = _hist.Histogram if histogram is None else histogram  # type: ignore

    if normed is not None:
        raise KeyError(
            "normed=True is not recommended for use in Numpy, and is not supported in boost-histogram; use density=True instead"
        )
    if density and histogram is not None:
        raise KeyError(
            "boost-histogram does not support the density keyword when returning a boost-histogram object"
        )

    # Odd NumPy design here. Oh well.
    if isinstance(a, np.ndarray):
        a = a.T

    rank = len(a)

    # Integer bins: all the same
    try:
        bins = (int(bins),) * rank  # type: ignore
    except TypeError:
        pass
    assert not isinstance(bins, int)

    # Single None -> list of Nones
    if range is None:
        range = (None,) * rank

    axs = []
    for n, (b, r) in enumerate(zip(bins, range)):
        if np.issubdtype(type(b), np.integer):
            if r is None:
                # Nextafter may affect bin edges slightly
                r = (np.min(a[n]), np.max(a[n]))
            cpp_ax = _core.axis.regular_numpy(b, r[0], r[1])
            new_ax = _cast(None, cpp_ax, _axis.Axis)
            axs.append(new_ax)
        else:
            barr = np.asarray(b, dtype=np.double)
            barr[-1] = np.nextafter(barr[-1], np.finfo("d").max)
            axs.append(_axis.Variable(barr))

    hist = cls(*axs, storage=storage).fill(*a, weight=weights, threads=threads)

    if density:
        areas = _reduce(_mul, hist.axes.widths)
        density_val = hist.values() / np.sum(hist.values()) / areas
        return (density_val, hist.to_numpy()[1:])

    # Note: this is view=True since users have to ask explicitly for special
    # storages, so view=False would throw away part of what they are asking
    # for. Users can use a histogram return type if they need view=False.
    return hist if histogram is not None else hist.to_numpy(view=True, dd=True)


def histogram2d(
    x: ArrayLike,
    y: ArrayLike,
    bins: Union[int, Tuple[int, int]] = 10,
    range: Optional[Sequence[Union[None, Tuple[float, float]]]] = None,
    normed: None = None,
    weights: Optional[ArrayLike] = None,
    density: bool = False,
    *,
    histogram: Optional[Type[_hist.Histogram]] = None,
    storage: _storage.Storage = _storage.Double(),  # noqa: B008
    threads: Optional[int] = None
) -> Any:
    result = histogramdd(
        (x, y),
        bins,
        range,
        normed,
        weights,
        density,
        histogram=histogram,
        storage=storage,
        threads=threads,
    )

    if not isinstance(result, tuple):
        return result

    data, (edgesx, edgesy) = result
    return data, edgesx, edgesy


def histogram(
    a: ArrayLike,
    bins: int = 10,
    range: Optional[Tuple[float, float]] = None,
    normed: None = None,
    weights: Optional[ArrayLike] = None,
    density: bool = False,
    *,
    histogram: Optional[Type[_hist.Histogram]] = None,
    storage: Optional[_storage.Storage] = None,
    threads: Optional[int] = None
) -> Any:
    np = _np

    # numpy 1d histogram returns integers in some cases
    if storage is None:
        storage = (
            _storage.Double()
            if weights is not None or normed or density
            else _storage.Int64()
        )

    if isinstance(bins, str):
        # Bug in NumPy 1.20 typing support - __version__ is missing
        if tuple(int(x) for x in np.version.version.split(".")[:2]) < (1, 13):
            raise KeyError(
                "Upgrade numpy to 1.13+ to use string arguments to boost-histogram's histogram function"
            )
        bins = np.histogram_bin_edges(a, bins, range, weights)

    result = histogramdd(
        (a,),
        (bins,),
        (range,),
        normed,
        weights,
        density,
        histogram=histogram,
        storage=storage,
        threads=threads,
    )
    if not isinstance(result, tuple):
        return result

    data, (edges,) = result
    return data, edges


# Process docstrings
for f, n in zip(
    (histogram, histogram2d, histogramdd),
    (_np.histogram, _np.histogram2d, _np.histogramdd),
):

    H = """\
    Return a boost-histogram object using the same arguments as numpy's {}.
    This does not support the deprecated normed=True argument. Three extra
    arguments are added: histogram=bh.Histogram will enable object based
    output, storage=bh.storage.* lets you set the storage used, and threads=int
    lets you set the number of threads to fill with (0 for auto, None for 1).
    """

    f.__doc__ = H.format(n.__name__) + n.__doc__

del f, n, H
