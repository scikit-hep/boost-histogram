from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from ._internal.hist import Histogram

from . import axis, storage, accumulators, algorithm, numpy

from .tag import loc, rebin, sum, underflow, overflow

# Workarounds for smooth transitions from 0.5 series. Will be removed after 0.6.
project = sum
histogram = Histogram

from .version import __version__
