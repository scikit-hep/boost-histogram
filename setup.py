# -*- coding: utf-8 -*-
from __future__ import division
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import setuptools
import sys


# Use -j N or set the environment variable NPY_NUM_BUILD_JOBS
try:
    from numpy.distutils.ccompiler import CCompiler_compile
    import distutils.ccompiler

    distutils.ccompiler.CCompiler.compile = CCompiler_compile
except ImportError:
    print("Numpy not found, parallel compile not available")

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
    Extension(
        "boost_histogram._core", SRC_FILES, include_dirs=INCLUDE_DIRS, language="c++"
    )
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++14 compiler flag.
    """
    if has_flag(compiler, "-std=c++14"):
        return "-std=c++14"
    else:
        raise RuntimeError("Unsupported compiler -- at least C++14 support is needed!")


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {"msvc": ["/EHsc", "/bigobj"], "unix": ["-g0"]}

    if sys.platform == "darwin":
        c_opts["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.9"]

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


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

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    install_requires=["numpy"],
    extras_require=extras,
)
