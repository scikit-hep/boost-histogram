from __future__ import annotations

import contextlib
import importlib.metadata
import sys
import sysconfig
from pathlib import Path

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib

import pytest
from packaging.requirements import Requirement

import boost_histogram as bh

DIR = Path(__file__).parent.resolve()
BASE = DIR.parent


@pytest.fixture(params=(False, True), ids=("no_growth", "growth"))
def growth(request):
    return request.param


@pytest.fixture(params=(False, True), ids=("no_overflow", "overflow"))
def overflow(request):
    return request.param


@pytest.fixture(params=(False, True), ids=("no_underflow", "underflow"))
def underflow(request):
    return request.param


@pytest.fixture(params=(False, True), ids=("no_flow", "flow"))
def flow(request):
    return request.param


@pytest.fixture(
    params=(None, "str", 1, {"a": 1}),
    ids=("no_metadata", "str_metadata", "int_metadata", "dict_metadata"),
)
def metadata(request):
    return request.param


@pytest.fixture(
    params=(
        bh.storage.Double,
        bh.storage.Int64,
        bh.storage.AtomicInt64,
        bh.storage.Weight,
        bh.storage.Unlimited,
    ),
    ids=("Double", "Int64", "AtomicInt64", "Weight", "Unlimited"),
)
def count_storage(request):
    return request.param


@pytest.fixture(
    params=(
        bh.storage.Double,
        bh.storage.Int64,
        bh.storage.AtomicInt64,
        bh.storage.Unlimited,
    ),
    ids=("Double", "Int64", "AtomicInt64", "Unlimited"),
)
def count_single_storage(request):
    return request.param


def pytest_report_header() -> str:
    with BASE.joinpath("pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    project = pyproject.get("project", {})

    pkgs = project.get("dependencies", [])
    pkgs += [p for ps in project.get("optional-dependencies", {}).values() for p in ps]
    if "name" in project:
        pkgs.append(project["name"])
    interesting_packages = {Requirement(p).name for p in pkgs}
    interesting_packages.add("pip")
    interesting_packages.add("numpy")

    valid = []
    for package in sorted(interesting_packages):
        with contextlib.suppress(ModuleNotFoundError):
            valid.append(f"{package}=={importlib.metadata.version(package)}")
    reqs = " ".join(valid)
    lines = [
        f"installed packages of interest: {reqs}",
    ]
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        lines.append("free-threaded Python build")
    return "\n".join(lines)
