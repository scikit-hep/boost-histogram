from __future__ import annotations

import contextlib
import typing
from collections.abc import Sequence
from functools import reduce
from operator import mul
from typing import Any

if typing.TYPE_CHECKING:
    from numpy.typing import ArrayLike
else:
    ArrayLike = object

import numpy as np

from . import axis as _axis
from . import storage as _storage
from .histogram import Histogram

__all__ = ("histogram", "histogram2d", "histogramdd")

# pylint: disable=redefined-builtin
# pylint: disable=redefined-outer-name


def __dir__() -> list[str]:
    return list(__all__)


def _outer_edges(a: ArrayLike, rng: tuple[float, float] | None) -> tuple[float, float]:
    """
    Compute the outer bin edges, following ``numpy.lib._histograms_impl._get_outer_edges``.
    """
    if rng is not None:
        first_edge, last_edge = rng
        if first_edge > last_edge:
            raise ValueError("max must be larger than min in range parameter.")
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                f"supplied range of [{first_edge}, {last_edge}] is not finite"
            )
    elif np.size(a) == 0:
        # No data, so no range can be found; NumPy uses 0-1 here.
        first_edge, last_edge = 0.0, 1.0
    else:
        first_edge, last_edge = np.amin(a), np.amax(a)
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                f"autodetected range of [{first_edge}, {last_edge}] is not finite"
            )

    # Expand an empty range to avoid a division by zero
    if first_edge == last_edge:
        first_edge -= 0.5
        last_edge += 0.5

    return first_edge, last_edge


def histogramdd(
    a: tuple[ArrayLike, ...],
    bins: int | tuple[int, ...] | tuple[np.typing.NDArray[Any], ...] = 10,
    range: Sequence[tuple[float, float] | None] | None = None,
    normed: None = None,
    weights: ArrayLike | None = None,
    density: bool = False,
    *,
    histogram: type[Histogram[Any]] | None = None,
    storage: _storage.Storage | None = None,
    threads: int | None = None,
) -> Any:
    if storage is None:
        storage = _storage.Double()
    cls: type[Histogram[Any]] = Histogram if histogram is None else histogram

    if normed is not None:
        raise KeyError(
            "normed=True is not recommended for use in NumPy, and is not supported in boost-histogram; use density=True instead"
        )
    if density and histogram is not None:
        raise KeyError(
            "boost-histogram does not support the density keyword when returning a boost-histogram object"
        )

    # Odd NumPy design here. Oh well. A 1D array is N samples of one dimension,
    # while an ND array has one sample per row.
    if isinstance(a, np.ndarray):  # type: ignore[unreachable]
        a = np.atleast_2d(a) if a.ndim == 1 else a.T  # type: ignore[unreachable]
    elif len(a) > 0 and np.ndim(a[0]) == 0:
        a = (np.asarray(a),)

    rank = len(a)

    # Integer bins: all the same
    with contextlib.suppress(TypeError):
        bins = (int(bins),) * rank  # type: ignore[arg-type]
    assert not isinstance(bins, int)
    if len(bins) != rank:
        raise ValueError(
            "The dimension of bins must be equal to the dimension of the sample x."
        )

    # Single None -> list of Nones
    if range is None:
        range = (None,) * rank
    elif len(range) != rank:
        raise ValueError("range argument must have one entry per dimension")

    axs: list[_axis.Axis] = []
    for n, (b, r) in enumerate(zip(bins, range, strict=True)):
        if np.issubdtype(type(b), np.integer):
            start, stop = _outer_edges(a[n], r)
            new_ax = _axis.Regular(
                typing.cast(int, b), start, stop, underflow=False, overflow=False
            )
            axs.append(new_ax)
        else:
            # Always copy; the input edges must not be modified in-place below
            barr: np.typing.NDArray[Any] = np.array(b, dtype=np.double)
            # This does have a .max member, not sure why pylint doesn't like it
            # pylint: disable-next=no-member
            barr[-1] = np.nextafter(barr[-1], np.finfo("d").max)
            axs.append(_axis.Variable(barr))

    hist = cls(*axs, storage=storage).fill(*a, weight=weights, threads=threads)

    if density:
        areas = reduce(mul, hist.axes.widths)
        density_val = hist.values() / np.sum(hist.values()) / areas
        return (density_val, hist.to_numpy()[1:])

    # Note: this is view=True since users have to ask explicitly for special
    # storages, so view=False would throw away part of what they are asking
    # for. Users can use a histogram return type if they need view=False.
    return hist if histogram is not None else hist.to_numpy(view=True, dd=True)


def histogram2d(
    x: ArrayLike,
    y: ArrayLike,
    bins: int | tuple[int, int] = 10,
    range: Sequence[tuple[float, float] | None] | None = None,
    normed: None = None,
    weights: ArrayLike | None = None,
    density: bool = False,
    *,
    histogram: type[Histogram[Any]] | None = None,
    storage: _storage.Storage | None = None,
    threads: int | None = None,
) -> Any:
    if storage is None:
        storage = _storage.Double()

    # NumPy shares a single edge array between both axes, but only if it is not
    # one or two entries long (those are read as per-axis bin counts).
    try:
        n_bins = len(bins)  # type: ignore[arg-type]
    except TypeError:
        n_bins = 1
    if n_bins not in {1, 2}:
        edges = np.asarray(bins)
        bins = (edges, edges)  # type: ignore[assignment]

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
    bins: int | str | np.typing.NDArray[Any] = 10,
    range: tuple[float, float] | None = None,
    normed: None = None,
    weights: ArrayLike | None = None,
    density: bool = False,
    *,
    histogram: type[Histogram[Any]] | None = None,
    storage: _storage.Storage | None = None,
    threads: int | None = None,
) -> Any:
    # numpy 1d histogram returns integers in some cases
    if storage is None:
        storage = (
            _storage.Double()
            if weights is not None or normed or density  # type: ignore[redundant-expr]
            else _storage.Int64()
        )

    if isinstance(bins, str):
        bins = np.histogram_bin_edges(a, bins, range, weights)

    # TODO: make sure all types work at runtime (type ignore below)
    # I think it's safe and the union is in the wrong place
    result = histogramdd(
        (a,),
        (bins,),  # type: ignore[arg-type]
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
# TODO: make this a decorator
for f, np_f in zip(
    (histogram, histogram2d, histogramdd),
    (np.histogram, np.histogram2d, np.histogramdd),
    strict=False,
):
    H = """\
    Return a boost-histogram object using the same arguments as numpy's {}.
    This does not support the deprecated normed=True argument. Three extra
    arguments are added: histogram=bh.Histogram will enable object based
    output, storage=bh.storage.* lets you set the storage used, and threads=int
    lets you set the number of threads to fill with (0 for auto, None for 1).
    """

    f.__doc__ = H.format(np_f.__name__) + (np_f.__doc__ or "")
