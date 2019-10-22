from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from ._hist import histogram

from . import axis, storage, accumulators, algorithm, numpy

from .uhi import loc, rebin, project, underflow, overflow

from .version import __version__
