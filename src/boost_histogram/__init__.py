from __future__ import absolute_import, division, print_function

# Sadly, some tools (IPython) do not respect __all__
# as a list of public items in a module. So we need
# to delete / hide any extra items manually.
del absolute_import, division, print_function

__all__ = (
    "Histogram",
    "axis",
    "storage",
    "accumulators",
    "utils",
    "numpy",
    "loc",
    "rebin",
    "sum",
    "underflow",
    "overflow",
    "__version__",
)


try:
    from . import _core
except ImportError as err:
    if "_core" in err.msg and "boost_histogram" in err.msg:
        err.msg += "\nDid you forget to compile? Use CMake or Setuptools to build, see the readme"
    raise err


from ._internal.hist import Histogram
from . import axis, storage, accumulators, utils, numpy
from .tag import loc, rebin, sum, underflow, overflow

from .version import version as __version__
