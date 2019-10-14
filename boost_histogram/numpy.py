from __future__ import absolute_import, division, print_function

from . import axis as _axis
from . import _hist as _hist
from . import core as _core

import warnings

warnings.warn(
    "The boost_histogram.numpy module is provisional and may change in future releases",
    FutureWarning,
)


def histogramdd(a, bins=10, range=None, normed=None, weights=None, density=None):
    import numpy as np

    if normed is not None:
        raise KeyError(
            "normed is not recommended for use in Numpy, and is not supported in boost-histogram; use density instead"
        )
    if density is not None:
        raise KeyError(
            "boost-histogram does not support the density keyword at the moment"
        )

    # Odd design here. Oh well.
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
        return _hist.histogram(*axs).fill(*a)
    else:
        return _hist.histogram(*axis).fill(*a, weight=weights)


def histogram2d(x, y, bins=10, range=None, normed=None, weights=None, density=None):
    return histogramdd((x, y), bins, range, normed, weights, density)


def histogram(x, bins=10, range=None, normed=None, weights=None, density=None):
    import numpy as np

    if isinstance(bins, str):
        if tuple(int(x) for x in np.__version__.split(".")[:2]) < (1, 13):
            raise KeyError(
                "Upgrade numpy to 1.13+ to use string arguments to boost-histogram's histogram function"
            )
        bins = np.histogram_bin_edges(x, bins, range, weights)
    return histogramdd((x,), (bins,), (range,), normed, weights, density)
    # TODO: this also supports a few more tricks in Numpy, those can be added for numpy 1.13+
