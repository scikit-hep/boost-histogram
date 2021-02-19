# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from . import accumulators, axis, numpy, storage, utils
from ._internal.enum import Kind
from ._internal.hist import Histogram
from .tag import loc, overflow, rebin, sum, underflow
from .version import version as __version__

try:
    from . import _core
except ImportError as err:
    msg = str(err)
    if "_core" not in msg:
        raise

    import sys

    new_msg = "Did you forget to compile boost-histogram? Use CMake or Setuptools to build, see the readme."
    total_msg = "\n".join([msg, new_msg])

    # Python 2
    if sys.version_info < (3, 0):
        orig = sys.exc_info()
        assert orig[0] is not None and orig[2] is not None
        exc_info = orig[0], orig[0](total_msg), orig[2]
        exec("raise exc_info[0], exc_info[1], exc_info[2]")
    else:
        new_exception = type(err)(new_msg, name=err.name, path=err.path)
        exec("raise new_exception from err")


# Sadly, some tools (IPython) do not respect __all__
# as a list of public items in a module. So we need
# to delete / hide any extra items manually.
del absolute_import, division, print_function

__all__ = (
    "Histogram",
    "Kind",
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


# Support cloudpickle - pybind11 submodules do not have __file__ attributes
# And setting this in C++ causes a segfault
_core.accumulators.__file__ = _core.__file__
_core.algorithm.__file__ = _core.__file__
_core.axis.__file__ = _core.__file__
_core.axis.transform.__file__ = _core.__file__
_core.hist.__file__ = _core.__file__
_core.storage.__file__ = _core.__file__
