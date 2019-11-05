from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from ._internal.hist import Histogram
from . import axis, storage, accumulators, algorithm, numpy
from .tag import loc, rebin, sum, underflow, overflow
from .version import __version__

# Workarounds for smooth transitions from 0.5 series. Will be removed after 0.6.
histogram = Histogram

from .version import __version__

from .tag import project


class histogram(Histogram):
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn("Use Histogram instead", DeprecationWarning)
        super(histogram, self).__init__(*args, **kwargs)
