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


def histogramdd(
    a: tuple[ArrayLike, ...],
    bins: int | tuple[int, ...] | tuple[np.typing.NDArray[Any], ...] = 10,
    range: None | (Sequence[None | tuple[float, float]]) = None,
    normed: None = None,
    weights: ArrayLike | None = None,
    density: bool = False,
    *,
    histogram: None | (type[Histogram]) = None,
    storage: _storage.Storage = _storage.Double(),  # noqa: B008
    threads: int | None = None,
) -> Any:
    cls: type[Histogram] = Histogram if histogram is None else histogram

    if normed is not None:
        raise KeyError(
            "normed=True is not recommended for use in NumPy, and is not supported in boost-histogram; use density=True instead"
        )
    if density and histogram is not None:
        raise KeyError(
            "boost-histogram does not support the density keyword when returning a boost-histogram object"
        )

    # Odd NumPy design here. Oh well.
    if isinstance(a, np.ndarray):  # type: ignore[unreachable]
        a = a.T  # type: ignore[unreachable]

    rank = len(a)

    # Integer bins: all the same
    with contextlib.suppress(TypeError):
        bins = (int(bins),) * rank  # type: ignore[arg-type]
    assert not isinstance(bins, int)

    # Single None -> list of Nones
    if range is None:
        range = (None,) * rank

    axs: list[_axis.Axis] = []
    for n, (b, r) in enumerate(zip(bins, range)):
        if np.issubdtype(type(b), np.integer):
            if r is None:
                # Nextafter may affect bin edges slightly
                r = (np.amin(a[n]), np.amax(a[n]))  # noqa: PLW2901
                if r[0] == r[1]:  # type: ignore[operator]
                    r = (r[0] - 0.5, r[1] + 0.5)  # noqa: PLW2901
            new_ax = _axis.Regular(
                typing.cast(int, b), r[0], r[1], underflow=False, overflow=False
            )
            axs.append(new_ax)
        else:
            barr: np.typing.NDArray[Any] = np.asarray(b, dtype=np.double)
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
    range: None | (Sequence[None | tuple[float, float]]) = None,
    normed: None = None,
    weights: ArrayLike | None = None,
    density: bool = False,
    *,
    histogram: None | (type[Histogram]) = None,
    storage: _storage.Storage = _storage.Double(),  # noqa: B008
    threads: int | None = None,
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
    bins: int | str | np.typing.NDArray[Any] = 10,
    range: tuple[float, float] | None = None,
    normed: None = None,
    weights: ArrayLike | None = None,
    density: bool = False,
    *,
    histogram: None | (type[Histogram]) = None,
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
        # Bug in NumPy 1.20 typing support - __version__ is missing
        if tuple(int(x) for x in np.version.version.split(".")[:2]) < (1, 13):
            raise KeyError(
                "Upgrade numpy to 1.13+ to use string arguments to boost-histogram's histogram function"
            )
        bins = np.histogram_bin_edges(a, bins, range, weights)

    # TODO: make sure all types work at runtime (type ignore below)
    # I think it's safe and the union is in the wrong place
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
# TODO: make this a decorator
for f, np_f in zip(
    (histogram, histogram2d, histogramdd),
    (np.histogram, np.histogram2d, np.histogramdd),
):
    H = """\
    Return a boost-histogram object using the same arguments as numpy's {}.
    This does not support the deprecated normed=True argument. Three extra
    arguments are added: histogram=bh.Histogram will enable object based
    output, storage=bh.storage.* lets you set the storage used, and threads=int
    lets you set the number of threads to fill with (0 for auto, None for 1).
    """

    f.__doc__ = H.format(np_f.__name__) + (np_f.__doc__ or "")
