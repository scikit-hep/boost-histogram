import os
import sys
from pathlib import Path

from setuptools import setup

DIR = Path(__file__).parent.resolve()

sys.path.append(str(DIR / "extern" / "pybind11"))
from pybind11.setup_helpers import ParallelCompile, Pybind11Extension  # noqa: E402

del sys.path[-1]

# Use the environment variable CMAKE_BUILD_PARALLEL_LEVEL to control parallel builds
ParallelCompile("CMAKE_BUILD_PARALLEL_LEVEL").install()

cxx_std = int(os.environ.get("CMAKE_CXX_STANDARD", "14"))

SRC_FILES = [
    "src/module.cpp",
    "src/register_accumulators.cpp",
    "src/register_algorithm.cpp",
    "src/register_axis.cpp",
    "src/register_histograms.cpp",
    "src/register_storage.cpp",
    "src/register_transforms.cpp",
]

INCLUDE_DIRS = [
    "include",
    "extern/assert/include",
    "extern/config/include",
    "extern/core/include",
    "extern/histogram/include",
    "extern/mp11/include",
    "extern/throw_exception/include",
    "extern/variant2/include",
]

ext_modules = [
    Pybind11Extension(
        "boost_histogram._core",
        SRC_FILES,
        include_dirs=INCLUDE_DIRS,
        cxx_std=cxx_std,
        extra_compile_args=["/d2FH4-"] if sys.platform.startswith("win32") else [],
    )
]


extras = {
    "test": ["pytest", "pytest-benchmark", "cloudpickle", "hypothesis >=6.0"],
    "docs": [
        "Sphinx~=3.0",
        "myst_parser>=0.13",
        "sphinx-book-theme>=0.0.33",
        "nbsphinx",
        "sphinx_copybutton",
    ],
    "examples": ["matplotlib", "xarray", "xhistogram", "netCDF4", "numba", "uproot3"],
    "dev": ["ipykernel", "typer"],
}
extras["all"] = sum(extras.values(), [])
extras["dev"] += extras["test"]

setup(ext_modules=ext_modules, extras_require=extras)
