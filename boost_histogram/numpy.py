from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function  # hides these from IPython

from . import axis as _axis
from . import _hist as _hist
from . import core as _core

from ._utils import KWArgs as _KWArgs

import numpy as _np

__all__ = ("histogram", "histogram2d", "histogramdd")


def histogramdd(
    a, bins=10, range=None, normed=None, weights=None, density=None, **kwargs
):
    np = _np  # Hidden to keep module clean

    with _KWArgs(kwargs) as k:
        boost = k.optional("bh", False)
        storage = k.optional("bh_storage", _core.storage.double)

    if normed is not None:
        raise KeyError(
            "normed is not recommended for use in Numpy, and is not supported in boost-histogram; use density instead"
        )
    if density is not None:
        raise KeyError(
            "boost-histogram does not support the density keyword at the moment"
        )

    # Odd numpy design here. Oh well.
    if isinstance(a, np.ndarray):
        a = a.T

    rank = len(a)

    # Integer bins: all the same
    try:
        bins = [int(bins)] * rank
    except TypeError:
        pass

    # Single None -> list of Nones
    if range is None:
        range = [None] * rank

    axs = []
    for n, (b, r) in enumerate(zip(bins, range)):
        if isinstance(b, int):
            if r is None:
                # Nextafter may affect bin edges slightly
                r = (np.min(a[n]), np.max(a[n]))
            axs.append(_core.axis._regular_numpy(b, r[0], r[1]))
        else:
            b = np.asarray(b, dtype=np.double)
            b[-1] = np.nextafter(b[-1], np.finfo("d").max)
            axs.append(_axis.variable(b))

    if weights is None:
        hist = _hist.histogram(*axs).fill(*a)
    else:
        hist = _hist.histogram(*axis).fill(*a, weight=weights)

    return hist if boost else hist.to_numpy()


def histogram2d(
    x, y, bins=10, range=None, normed=None, weights=None, density=None, **kwargs
):
    return histogramdd((x, y), bins, range, normed, weights, density, **kwargs)


def histogram(
    a, bins=10, range=None, normed=None, weights=None, density=None, **kwargs
):
    np = _np

    # numpy 1d histogram returns integers in some cases
    if "bh_storage" not in kwargs and not (weights or normed or density):
        kwargs["bh_storage"] = _core.storage.int

    if isinstance(bins, str):
        if tuple(int(x) for x in np.__version__.split(".")[:2]) < (1, 13):
            raise KeyError(
                "Upgrade numpy to 1.13+ to use string arguments to boost-histogram's histogram function"
            )
        bins = np.histogram_bin_edges(a, bins, range, weights)
    return histogramdd((a,), (bins,), (range,), normed, weights, density, **kwargs)


# Process docstrings
for f, n in zip(
    (histogram, histogram2d, histogramdd),
    (_np.histogram, _np.histogram2d, _np.histogramdd),
):

    H = """\
    Return a boost-histogram object using the same arguments as numpy's {}.
    This does not support density/normed yet. Two extra arguments are added: bh=True
    will enable object based output, and bh_storage=... lets you set the storage used.
    """

    f.__doc__ = H.format(n.__name__) + n.__doc__

del f, n, H
