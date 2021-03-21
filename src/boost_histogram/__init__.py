from . import accumulators, axis, numpy, storage
from ._internal.enum import Kind
from ._internal.hist import Histogram, IndexingExpr
from .tag import loc, overflow, rebin, sum, underflow
from .version import version as __version__

try:
    from . import _core
except ImportError as err:
    msg = str(err)
    if "_core" not in msg:
        raise

    new_msg = "Did you forget to compile boost-histogram? Use CMake or Setuptools to build, see the readme."
    total_msg = f"{msg}\n{new_msg}"

    new_exception = type(err)(new_msg, name=err.name, path=err.path)
    raise new_exception from err


__all__ = (
    "Histogram",
    "IndexingExpr",
    "Kind",
    "axis",
    "storage",
    "accumulators",
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
