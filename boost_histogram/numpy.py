from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function  # hides these from IPython

__all__ = ("histogram", "histogram2d", "histogramdd")

from . import axis as _axis
from ._internal import hist as _hist
from ._internal.utils import cast as _cast
from . import _core
from . import storage as _storage

from ._internal.kwargs import KWArgs as _KWArgs

import numpy as _np


def histogramdd(
    a, bins=10, range=None, normed=None, weights=None, density=None, **kwargs
):
    np = _np  # Hidden to keep module clean

    with _KWArgs(kwargs) as k:
        bh_warning = k.optional("bh")
        if bh_warning is not None:
            import warnings

            warnings.warn(
                "bh=True has been replaced by histogram=bh.Histogram", FutureWarning
            )
            bh_warning = bh.Histogram
        else:
            bh_warning = None
        bh_cls = k.optional("histogram", bh_warning)
        cls = _hist.Histogram if bh_cls is None else bh_cls
        bh_storage = k.optional("storage", _storage.Double())

    if normed is not None:
        raise KeyError(
            "normed=True is not recommended for use in Numpy, and is not supported in boost-histogram; use density=True instead"
        )
    if density and bh_cls is not None:
        raise KeyError(
            "boost-histogram does not support the density keyword when returning a boost-histogram object"
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
            cpp_ax = _core.axis.regular_numpy(b, r[0], r[1], None)
            new_ax = _cast(None, cpp_ax, _axis.Axis)
            axs.append(new_ax)
        else:
            b = np.asarray(b, dtype=np.double)
            b[-1] = np.nextafter(b[-1], np.finfo("d").max)
            axs.append(_axis.Variable(b))

    hist = cls(*axs, storage=bh_storage).fill(*a, weight=weights)

    if density:
        areas = np.prod(hist.axes.widths, axis=0)
        density = hist.view() / hist.sum() / areas
        return (density,) + hist.to_numpy()[1:]

    return hist if bh_cls is not None else hist.to_numpy()


def histogram2d(
    x, y, bins=10, range=None, normed=None, weights=None, density=None, **kwargs
):
    return histogramdd((x, y), bins, range, normed, weights, density, **kwargs)


def histogram(
    a, bins=10, range=None, normed=None, weights=None, density=None, **kwargs
):
    np = _np

    # numpy 1d histogram returns integers in some cases
    if "storage" not in kwargs and not (weights or normed or density):
        kwargs["storage"] = _storage.Int64()

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
    This does not support the deprecated normed=True argument. Two extra
    arguments are added: histogram=bh.Histogram will enable object based
    output, and storage=bh.storage.* lets you set the storage used.
    """

    f.__doc__ = H.format(n.__name__) + n.__doc__

del f, n, H
