from __future__ import annotations

import sys

try:
    from . import _core  # noqa: F401
except ImportError as err:
    if "_core" not in str(err):
        raise

    new_msg = "Did you forget to compile boost-histogram? Use CMake or scikit-build-core to build, see the readme."

    if sys.version_info >= (3, 11):
        err.add_note(new_msg)
        raise

    total_msg = f"{err}\n{new_msg}"
    new_exception = type(err)(total_msg, name=err.name, path=err.path)
    raise new_exception from err

from . import accumulators, axis, numpy, storage
from .histogram import Histogram, IndexingExpr, Kind
from .tag import (  # pylint: disable=redefined-builtin
    loc,
    overflow,
    rebin,
    sum,
    underflow,
)

# pylint: disable-next=import-error
from .version import version as __version__

__all__ = [
    "Histogram",
    "IndexingExpr",
    "Kind",
    "__version__",
    "accumulators",
    "axis",
    "loc",
    "numpy",
    "overflow",
    "rebin",
    "storage",
    "sum",
    "underflow",
]


def __dir__() -> list[str]:
    return __all__
