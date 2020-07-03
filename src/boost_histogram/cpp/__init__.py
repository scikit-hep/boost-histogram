# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import warnings

from .._internal.hist import histogram

from . import axis, storage, accumulators, algorithm
from .. import __version__

del absolute_import, division, print_function

msg = "The cpp module has been deprecated, and will be removed after 0.8.0"
warnings.warn(msg, FutureWarning)

__all__ = ("histogram", "axis", "storage", "accumulators", "algorithm", "__version__")
