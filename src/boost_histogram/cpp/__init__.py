# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from .._internal.hist import histogram

from . import axis, storage, accumulators, algorithm
from .. import __version__

del absolute_import, division, print_function

__all__ = ("histogram", "axis", "storage", "accumulators", "algorithm", "__version__")
