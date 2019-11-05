from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from .._internal.hist import BoostHistogram as histogram

from . import axis, storage, accumulators, algorithm
from ..version import __version__
