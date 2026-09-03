# CLAUDE.md

This file provides guidance for agents when working with code in this repository.

## What this is

Python bindings for [Boost.Histogram](https://github.com/boostorg/histogram), a header-only C++14
histogramming library. This is a hybrid Python/C++ project: pybind11 wraps the C++ library into a
compiled `_core` extension module, and a pure-Python layer in `src/boost_histogram/` provides the
user-facing API on top of it.

## Commands

Prefer `uv run <cmd>` — it rebuilds the C++ extension as needed before running.

| Task                               | Command                                            |
| ---------------------------------- | -------------------------------------------------- |
| Install for dev                    | `uv sync` (or `pip install -ve. --group dev`)      |
| Run tests                          | `uv run pytest`                                    |
| Run one test                       | `uv run pytest tests/test_histogram.py::test_name` |
| Tests in parallel, no benchmark    | `uv run pytest -n auto --benchmark-disable`        |
| Lint / format / typecheck          | `prek -a --quiet`                                  |
| Type check only                    | `prek run mypy -a` (strict on `src/`, `examples/`) |
| Build docs (serves if interactive) | `nox -s docs`                                      |
| C++ (CMake) build + test           | `cmake --workflow default`                         |

`nox -l` lists all sessions. `prek` is used instead of `pre-commit run`.

The C++ build needs git submodules (`extern/`): `git submodule update --init --depth 10`.

## Testing notes

- `pytest-benchmark` is a required plugin; CI passes `--benchmark-disable`. Benchmark tests live in
  `tests/test_benchmark_*.py`.
- `filterwarnings = error` and `xfail_strict = true` — warnings fail tests, and unexpected xpasses fail.
- Property-based tests use Hypothesis (`tests/test_pbt.py`); some systems need `CI=1` to pass health checks.
- Python 3.14t (free-threading) is supported; thread-safety tests are in `tests/test_threaded_fill.py`.

## Architecture

### Two layers, bridged by `cast`/`register`

The C++ extension exposes raw types under `_core` submodules (`_core.hist`, `_core.axis`,
`_core.storage`, `_core.accumulators`, `_core.algorithm`, `_core.axis.transform`) — see
`src/module.cpp`. These raw C++ objects are wrapped by Python classes in `src/boost_histogram/`.

The bridge is in `src/boost_histogram/_utils.py`:

- `@register({cpp_type, ...})` decorates a Python class to claim ownership of one or more C++ types.
- `cast(self, cpp_object, ParentClass)` converts a raw C++ object into the right Python wrapper by
  walking subclasses of `ParentClass` and matching the registered C++ type. Classes that can't be
  constructed directly from the C++ object provide a `_convert_cpp` classmethod (pybind11 can't
  downcast — see the comment in `_cast_make_object`).
- The `_family` attribute lets downstream libraries (e.g. [Hist](https://github.com/scikit-hep/hist))
  subclass these wrappers and have `cast` return _their_ subclass instead. `boost_histogram` is the
  fallback family. This is why most wrapper classes are decorated and carry `_family`.

When adding a new axis/storage/accumulator wrapper, follow this pattern: register the C++ type, set
`_family`, and add a `_convert_cpp` if direct construction doesn't work.

### Key modules

- `histogram.py` — the `Histogram` class wrapping `_core.hist`, plus UHI indexing (`loc`, `rebin`,
  `sum`, `underflow`, `overflow` from `tag.py`), `__getitem__`/`__setitem__`, projection, views.
- `axis/__init__.py` — axis wrappers (`Regular`, `Variable`, `Integer`, `IntCategory`, `StrCategory`,
  `Boolean`) and `AxesTuple`; `axis/transform.py` — log/sqrt/pow/function transforms.
- `storage.py`, `accumulators.py`, `view.py` — storage types, accumulators, and NumPy structured-array
  views (`WeightedSumView`, `MeanView`, `WeightedMeanView`).
- `numpy.py` — drop-in replacements for `numpy.histogram*` functions.
- `serialization/` — pickle/serialization helpers for axes and storages.

### C++ side

- `src/register_*.cpp` — each registers one category (axes, storages, histograms, accumulators,
  transforms, algorithms) into the `_core` module defined in `src/module.cpp`.
- `include/bh_python/` — the binding headers (fill logic, pickling, metadata, NumPy interop, etc.).
- `extern/histogram/` and other `extern/*` are git submodules (Boost headers); don't edit them.
  `nox -s bump_boost -- <version>` updates them.
- `.pyi` stubs for the compiled module live in `src/boost_histogram/_core/`.

## Conventions

- `from __future__ import annotations` is required at the top of every Python file (ruff-enforced).
- Use `boost_histogram._compat.typing.Self`, not `typing.Self`/`typing_extensions.Self` (banned-api).
- mypy runs in strict mode on `src/` and `examples/`; tests are exempt from untyped-def rules.
- Version is dynamic from git tags via `setuptools_scm`, generated at build time into
  `boost_histogram/version.py` (do not commit/edit it).

## Supported Python versions

When the supported versions change (a new Python, a dropped one, a new cibuildwheel version, or a
new wheel platform), update all of these:

- `requires-python` and the `Programming Language :: Python :: *` classifiers in `pyproject.toml`
- the matrix in `.github/workflows/tests.yml` and the jobs in `.github/workflows/wheels.yml`
- the "Binaries available" wheel table in `README.md`

Get the true list of wheels from cibuildwheel instead of writing it by hand, one call per platform:

```bash
uvx cibuildwheel --platform linux --print-build-identifiers
```

Free-threaded and alternative-implementation builds are opt-in and change between cibuildwheel
releases, so a version can disappear from the table without an intentional drop.
