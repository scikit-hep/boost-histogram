# -*- coding: utf-8 -*-
from __future__ import division

from setuptools import setup

import sys
import os
import distutils.ccompiler


sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from setup_helpers import CppExtension  # noqa: E402

del sys.path[-1]


# Optional parallel compile utility
# inspired by: http://stackoverflow.com/questions/11013851/speeding-up-build-process-with-distutils
# and: https://github.com/tbenthompson/cppimport/blob/stable/cppimport/build_module.py
# and NumPy's parallel distutils module: https://github.com/numpy/numpy/blob/master/numpy/distutils/ccompiler.py
def make_parallel_compile(envvar=None, default=0, max=0):
    """
    Make a parallel compile function.

    envvar: Set an environment variable to control the compilation threads, like NPY_NUM_BUILD_JOBS
    default: 0 will automatically multithread, or 1 will only multithread if the envvar is set.
    max: The limit for automatic multithreading if non-zero

    To use:

        import distutils.ccompiler
        distutils.ccompiler.CCompiler.compile = make_parallel_compile("NPY_NUM_BUILD_JOBS")

    """

    def parallel_compile(
        self,
        sources,
        output_dir=None,
        macros=None,
        include_dirs=None,
        debug=0,
        extra_preargs=None,
        extra_postargs=None,
        depends=None,
    ):

        # These lines are copied directly from distutils.ccompiler.CCompiler
        macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
            output_dir, macros, include_dirs, sources, depends, extra_postargs
        )
        cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

        N = default

        # Determine the number of compilation threads, unless set by an environment variable.
        if envvar is not None:
            N = int(os.environ.get(envvar, default))

        def _single_compile(obj):
            try:
                src, ext = build[obj]
            except KeyError:
                return
            self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        try:
            import multiprocessing
            import multiprocessing.pool
        except ImportError:
            N = 1

        if N == 0:
            try:
                N = multiprocessing.cpu_count()
                N = max if max and max < N else N
            except NotImplementedError:
                N = 1

        # imap is evaluated on demand, converting to list() forces execution
        if N > 1:
            list(
                multiprocessing.pool.ThreadPool(N).imap_unordered(
                    _single_compile, objects
                )
            )
        else:
            for ob in objects:
                _single_compile(ob)

        return objects

    return parallel_compile


# Use the environment variable NPY_NUM_BUILD_JOBS

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
    "extern/pybind11/include",
    "extern/throw_exception/include",
    "extern/variant2/include",
]

ext_modules = [
    CppExtension(
        "boost_histogram._core", SRC_FILES, include_dirs=INCLUDE_DIRS, cxx_std=14
    )
]


extras = {
    "test": ["pytest", "pytest-benchmark"],
    "docs": [
        "Sphinx>=2.0.0",
        "recommonmark>=0.5.0",
        "sphinx_book_theme==0.30.0",
        "nbsphinx",
        "sphinx_copybutton",
    ],
    "examples": ["matplotlib", "xarray", "xhistogram", "netCDF4", "numba", "uproot"],
    "dev": ["ipykernel", "cloudpickle"],
}
extras["all"] = sum(extras.values(), [])
extras["dev"] += extras["test"]

distutils.ccompiler.CCompiler.compile = make_parallel_compile("NPY_NUM_BUILD_JOBS")

setup(
    ext_modules=ext_modules, extras_require=extras,
)
