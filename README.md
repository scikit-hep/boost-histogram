# boost-histogram for Python

[![Gitter][gitter-badge]][gitter-link]
[![Build Status][azure-badge]][azure-link]

Python bindings for [Boost::Histogram][], a C++14 library. This should become one of the [fastest libraries][] for histogramming, while still providing the power of a full histogram object.

> # Warning: This bindings are in progress and are not yet in an alpha stage.
>
> Join the discussion on gitter to follow the development!

[Boost::Histogram]:  https://www.boost.org/doc/libs/develop/libs/histogram/doc/html/index.html 
[fastest libraries]: https://iscinumpy.gitlab.io/post/histogram-speeds-in-python/


## Installation

This library is under development, but you can install directly from github if you would like. You need a C++14 compiler and Python 2.7--3.7.
All the normal best-practices for Python apply; you should be in a virtual environment, otherwise add `--user`, etc.

```bash
python -m pip install git+https://github.com/scikit-hep/boost-histogram.git@develop
```


## Developing

This repository has dependencies in submodules. Check out the repository like this:

```bash
git clone --recursive https://github.com/scikit-hep/boost-histogram.git
cd boost-histogram
```


<details><summary>Faster version (click to expand)</summary>

```bash
git clone https://github.com/scikit-hep/boost-histogram.git
cd boost-histogram
git submodule update --init --depth 10
```

</details>

Make a build directory and run CMake. If you have a specific Python you want to use, add `-DPYTHON_EXECUTABLE=$(which python)` or similar to the CMake line.

```bash
mkdir build
cd build
cmake ..
make -j4
```

Run the unit tests (requires pytest and numpy). Use `ctest` or `make test`, like this:

```bash
make test
```

[gitter-badge]: https://badges.gitter.im/HSF/PyHEP-histogramming.svg
[gitter-link]:  https://gitter.im/HSF/PyHEP-histogramming?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
[azure-badge]:  https://dev.azure.com/scikit-hep/boost-histogram/_apis/build/status/scikit-hep.boost-histogram?branchName=develop
[azure-link]:   https://dev.azure.com/scikit-hep/boost-histogram/_build/latest?definitionId=2&branchName=develop
