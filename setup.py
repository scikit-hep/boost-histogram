# -*- coding: utf-8 -*-
from __future__ import division

import os
import sys

from setuptools import setup

DIR = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(DIR, "extern", "pybind11"))
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
    "test": ["pytest", "pytest-benchmark", "typing_extensions", "cloudpickle"],
    "docs": [
        "Sphinx~=3.0",
        "recommonmark>=0.5.0",
        "sphinx_book_theme==0.38.0",
        "nbsphinx",
        "sphinx_copybutton",
    ],
    "examples": ["matplotlib", "xarray", "xhistogram", "netCDF4", "numba", "uproot3"],
    "dev": ["ipykernel", "typer"],
}
extras["all"] = sum(extras.values(), [])
extras["dev"] += extras["test"]

setup(ext_modules=ext_modules, extras_require=extras)
