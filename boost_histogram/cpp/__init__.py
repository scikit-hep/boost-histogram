from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

__all__ = ("histogram", "axis", "storage", "accumulators", "algorithm", "__version__")

from .._internal.hist import histogram

from . import axis, storage, accumulators, algorithm
from ..version import __version__
