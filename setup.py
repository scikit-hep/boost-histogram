import os
import sys
from pathlib import Path

from setuptools import setup

DIR = Path(__file__).parent.resolve()

sys.path.append(str(DIR / "extern" / "pybind11"))
from pybind11.setup_helpers import ParallelCompile, Pybind11Extension  # noqa: E402

del sys.path[-1]


def cross_compile_patch() -> None:  # noqa: C901
    import os

    target_arch = os.environ.get("TARGET_ARCH", None)
    if target_arch:
        import copy
        import distutils
        import distutils.sysconfig
        import sysconfig

        get_config_var_backup = sysconfig.get_config_var  # noqa: F841
        get_platform_backup = sysconfig.get_platform
        get_config_vars_backup = sysconfig.get_config_vars
        distutils_get_config_vars_backup = distutils.sysconfig.get_config_vars

        def get_platform():
            out = get_platform_backup()
            if target_arch is not None and isinstance(out, str):
                original_out = out
                out = out.replace("x86_64", target_arch)
                print("   Replace %s -> %s", original_out, out)
            return out

        def get_config_vars(*args):
            out = get_config_vars_backup(*args)
            target_arch = os.environ.get("TARGET_ARCH", None)
            if target_arch is None:
                return out
            out_xfix = copy.deepcopy(out)
            for k, v in out.items():
                if not (isinstance(v, str) and "x86_64" in v):
                    continue
                if k not in {"SO", "SOABI", "EXT_SUFFIX", "BUILD_GNU_TYPE"}:
                    continue
                fix = v.replace("x86_64", target_arch)
                print("   Replace %s: %s -> %s", k, v, fix)
                out_xfix[k] = fix
            return out_xfix

        def distutils_get_config_vars(*args):
            out = distutils_get_config_vars_backup(*args)
            target_arch = os.environ.get("TARGET_ARCH", None)
            if target_arch is None:
                return out
            if isinstance(out, list):
                fixes = []
                for item in out:
                    if not (isinstance(item, str) and "x86_64" in item):
                        fixes.append(item)
                    else:
                        fixes.append(item.replace("x86_64", target_arch))
                return fixes
            out_xfix = copy.deepcopy(out)
            for k, v in out.items():
                if not (isinstance(v, str) and "x86_64" in v):
                    continue
                if k not in {"SO", "SOABI", "EXT_SUFFIX", "BUILD_GNU_TYPE"}:
                    continue
                fix = v.replace("x86_64", target_arch)
                print("   Replace %s: %s -> %s", k, v, fix)
                out_xfix[k] = fix
            return out_xfix

        sysconfig.get_platform = get_platform
        sysconfig.get_config_vars = get_config_vars
        distutils.sysconfig.get_config_vars = distutils_get_config_vars


cross_compile_patch()

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
