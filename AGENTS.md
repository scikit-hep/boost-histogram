# boost-histogram Agent Guide

## Build System

**Hybrid Python/C++ project** using pybind11. Two build paths:

- **Python (recommended for most work):** `uv sync` or `pip install -ve .`
- **C++ development:** `cmake --preset default` then `cmake --build --preset default`

You can also prefix commands with `uv run` (recommended), this builds as needed.

Git submodules required for C++ build:
```bash
git clone --recursive https://github.com/scikit-hep/boost-histogram.git
# or if already cloned:
git submodule update --init --depth 10
```

## Commands

| Task | Command |
|------|---------|
| Lint, format, and typecheck | `prek -a` |
| Tests only | `uv run pytest` |
| Build docs + serve | `nox -s docs -- serve` |
| CMake build + test | `cmake --workflow default` |


## Project Structure

```
src/boost_histogram/    # Python package
src/*.cpp               # pybind11 bindings
extern/histogram/       # Boost.Histogram C++ headers (git submodule)
tests/                  # pytest suite
```

Key entry points:
- `src/boost_histogram/__init__.py` - Main Python API
- `src/module.cpp` - pybind11 module definition

## Testing Quirks

- Requires `pytest-benchmark` plugin (in `dev` dependency group)
- Use `--benchmark-disable` to skip benchmarks in CI
- Hypothesis tests may need `CI=1` env var on some systems to pass health checks
- Run single test: `python -m pytest tests/test_file.py::test_function`

## Toolchain Notes

- **Python:** 3.10+ required, 3.14t (free-threading) supported
- **C++:** C++14 standard, gcc >= 5.5, clang >= 3.8, or MSVC >= 14.1
- **Build backend:** scikit-build-core + pybind11
- **Formatter:** ruff (lint + format), clang-format (C++)
- **Type checker:** mypy (strict mode on src/)

## Versioning

- Dynamic version from git tags via `setuptools_scm`
- Version generated at build time in `boost_histogram/version.py`

## Related Projects

- [Hist](https://github.com/scikit-hep/hist) - High-level wrapper extending boost-histogram
- [UHI](https://uhi.readthedocs.io) - Histogram interoperability spec
- [mplhep](https://mplhep.readthedocs.io) - Plotting with UHI support
