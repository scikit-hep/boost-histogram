<img alt="boost-histogram logo" width="402" src="https://raw.githubusercontent.com/scikit-hep/boost-histogram/develop/docs/_images/BoostHistogramPythonLogo.png"/>

# boost-histogram for Python

[![Gitter][gitter-badge]][gitter-link]
[![Build Status][azure-badge]][azure-link]
[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![DOI](https://zenodo.org/badge/148885351.svg)](https://zenodo.org/badge/latestdoi/148885351)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![PyPI version](https://badge.fury.io/py/boost-histogram.svg)](https://pypi.org/project/boost-histogram/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/boost-histogram)][conda-link]

Python bindings for [Boost::Histogram][] ([source][Boost::Histogram source]), a C++14 library. This should become one of the [fastest libraries][] for histogramming, while still providing the power of a full histogram object.

> ## Version 0.6.2: Public beta
>
> Please feel free to try out boost-histogram and give feedback.
> Join the [discussion on gitter][gitter-link] or [open an issue](https://github.com/scikit-hep/boost-histogram/issues)!


## Installation

You can install this library from [PyPI](https://pypi.org/project/boost-histogram/) with pip:

```bash
python -m pip install boost-histogram
```

or you can use Conda through [conda-forge](https://github.com/conda-forge/boost-histogram-feedstock):

```bash
conda install -c conda-forge boost-histogram
```

All the normal best-practices for Python apply; you should be in a virtual environment, etc.


## Usage


```python
import boost_histogram as bh

# Compose axis however you like; this is a 2D histogram
hist = bh.Histogram(bh.axis.Regular(2, 0, 1),
                    bh.axis.Regular(4, 0.0, 1.0))

# Filling can be done with arrays, one per dimension
hist.fill([.3, .5, .2],
          [.1, .4, .9])

# Numpy array view into histogram counts, no overflow bins
counts = hist.view()
```

## Features

* Many axis types (all support `metadata=...`)
    * `bh.axis.Regular(n, start, stop, ...)`: Make a regular axis. Options listed below.
        * `overflow=False`: Turn off overflow bin
        * `underflow=False`: Turn off underflow bin
        * `growth=True`: Turn on growing axis, bins added when out-of-range items added
        * `circular=True`: Turn on wrapping, so that out-of-range values wrap around into the axis
        * `transform=bh.axis.transform.Log`: Log spacing
        * `transform=bh.axis.transform.Sqrt`: Square root spacing
        * `transform=bh.axis.transform.Pow(v)`: Power spacing
        * See also the flexible [Function transform](https://boost-histogram.readthedocs.io/en/latest/usage/transforms.html)
    * `bh.axis.Integer(start, stop, underflow=True, overflow=True, growth=False)`: Special high-speed version of `regular` for evenly spaced bins of width 1
    * `bh.axis.Variable([start, edge1, edge2, ..., stop], underflow=True, overflow=True)`: Uneven bin spacing
    * `bh.axis.Category([...], growth=False)`: Integer or string categories
* Axis features:
    * `.index(values)`: The index at a point (or points) on the axis
    * `.value(indexes)`: The value for a fractional bin in the axis
    * `.bin(i)`: The bin edges or a bin value (categories)
    * `.centers`: The N bin centers (if continuous)
    * `.edges`: The N+1 bin edges (if continuous)
    * `.extent`: The number of bins (including under/overflow)
    * `.metadata`: Anything a user wants to store
    * `.options`: The options set on the axis (`bh.axis.options`)
    * `.size`: The number of bins (not including under/overflow)
    * `.widths`: The N bin widths

* Many storage types
    * `bh.storage.Double()`: Doubles for weighted values (default)
    * `bh.storage.Int64()`: 64-bit unsigned integers
    * `bh.storage.Unlimited()`: Starts small, but can go up to unlimited precision ints or doubles.
    * `bh.storage.AtomicInt64()`: Threadsafe filling, experimental. Does not support growing axis in threads.
    * `bh.storage.Weight()`: Stores a weight and sum of weights squared.
    * `bh.storage.Mean()`: Accepts a sample and computes the mean of the samples (profile).
    * `bh.storage.WeightedMean()`: Accepts a sample and a weight. It computes the weighted mean of the samples.
* Accumulators
    * `bh.accumulator.Sum`: High accuracy sum (Neumaier) - used by the sum method when summing a numerical histogram
    * `bh.accumulator.WeightedSum`: Tracks a weighted sum and variance
    * `bh.accumulator.Mean`: Running count, mean, and variance (Welfords's incremental algorithm)
    * `bh.accumulator.WeightedMean`: Tracks a weighted sum, mean, and variance (West's incremental algorithm)
* Histogram operations
    * `h.fill(arr, ..., weight=...)` Fill with N arrays or single values
    * `h.rank`: The number of dimensions
    * `h.size or len(h)`: The number of bins
    * `.reset()`: Set counters to 0
    * `+`: Add two histograms
    * `*=`: Multiply by a scaler (not all storages) (`hist * scalar` and `scalar * hist` supported too)
    * `/=`: Divide by a scaler (not all storages) (`hist / scalar` supported too)
    * `.to_numpy(flow=False)`: Convert to a Numpy style tuple (with or without under/overflow bins)
    * `.view(flow=False)`: Get a view on the bin contents (with or without under/overflow bins)
    * `.axes`: Get the axes
        * `.axes[0]`: Get the 0th axis
        * `.axes.edges`: The lower values as a broadcasting-ready array
        * All other properties of axes available here, too
    * `.sum(flow=False)`: The total count of all bins
    * `.project(ax1, ax2, ...)`: Project down to listed axis (numbers)
* Indexing - Supports the [Unified Histogram Indexing (UHI)](https://boost-histogram.readthedocs.io/en/latest/usage/indexing.html) proposal
* Details
    * Use `bh.Histogram(..., storage=...)` to make a histogram (there are several different types)


## Supported platforms

#### Binaries available:

The easiest way to get boost-histogram is to use a binary wheel, which happens when you run:

```bash
python -m pip install boost-histogram
```

These are the supported platforms for which wheels are produced:

| System | Arch | Python versions |
|---------|-----|------------------|
| ManyLinux1 (custom GCC 9.2) | 64 & 32-bit | 2.7, 3.5, 3.6, 3.7, 3.8 |
| ManyLinux2010 | 64-bit | 2.7, 3.5, 3.6, 3.7, 3.8 |
| macOS 10.9+ | 64-bit | 2.7, 3.6, 3.7, 3.8 |
| Windows | 64 & 32-bit | 2.7, 3.6, 3.7, 3.8 |


* Linux: I'm not supporting 3.4 because I have to build the Numpy wheels to do so.
* manylinux1: Using a custom docker container with GCC 9.2; should work but can't be called directly other compiled extensions unless they do the same thing (think that's the main caveat). Supporting 32 bits because it's there.
* manylinux2010: Requires pip 10+ and a version of Linux newer than 2010. This is very new technology.
* MacOS: Uses the dedicated 64 bit 10.9+ Python.org builds. We are not supporting 3.5 because those no longer provide binaries (could add a 32+64 fat 10.6+ that really was 10.9+, but not worth it unless there is a need for it).
* Windows: PyBind11 requires compilation with a newer copy of Visual Studio than Python 2.7's Visual Studio 2008; you need to have the [Visual Studio 2015 distributable][msvc2015] installed (the dll is included in 2017 and 2019, as well).

[msvc2015]: https://www.microsoft.com/en-us/download/details.aspx?id=48145

If you are on a Linux system that is not part of the "many" in manylinux, such as Alpine or ClearLinux, building from source is usually fine, since the compilers on those systems are often quite new. It will just take a little longer to install when it's using the sdist instead of a wheel.


#### Conda-Forge

The boost-histogram package is available on Conda-Forge, as well. All supported versions are available with the exception of Windows + Python 2.7, which cannot built due to the age of the compiler. Please use Pip if you *really* need Python 2.7 on Windows. You will also need the VS 2015 distributable, as described above.

```
conda install -c conda-forge boost-histogram
```

#### Source builds

For a source build, for example from an "sdist" package, the only requirements are a C++14 compatible compiler. The compiler requirements are dictated by Boost.Histogram's C++ requirements: gcc >= 5.5, clang >= 3.8, msvc >= 14.1.

If you are using Python 2.7 on Windows, you will need to use a recent version of Visual studio and force distutils to use it, or just upgrade to Python 3.6 or newer. Check the PyBind11 documentation for [more help](https://pybind11.readthedocs.io/en/stable/faq.html#working-with-ancient-visual-studio-2009-builds-on-windows). On some Linux systems, you may need to use a newer compiler than the one your distribution ships with.

Having Numpy before building is recommended (enables multithreaded builds). Boost is not required or needed (this only depends on included header-only dependencies).This library is under active development; you can install directly from GitHub if you would like.

```bash
python -m pip install git+https://github.com/scikit-hep/boost-histogram.git@develop
```

For the moment, you need to uninstall and reinstall to ensure you have the latest version - pip will not rebuild if it thinks the version number has not changed. In the future, this may be addressed differently in boost-histogram.

## Developing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for details on how to set up a development environment.


## Talks and other documentation/tutorial sources

The [official documentation is here](https://boost-histogram.readthedocs.io/en/latest/index.html), and includes a [quickstart](https://boost-histogram.readthedocs.io/en/latest/usage/quickstart.html).

* [2019-4-15 IRIS-HEP Topical meeting](https://indico.cern.ch/event/803122/)
* [2019-10-17 PyHEP Histogram session](https://indico.cern.ch/event/833895/contributions/3577833/) - [repo with talks and workbook](https://github.com/henryiii/pres-bhandhist)
* [2019-11-7 CHEP](https://indico.cern.ch/event/773049/contributions/3473265/)

---

## Acknowledgements

This library was primarily developed by Henry Schreiner and Hans Dembinski.

Support for this work was provided by the National Science Foundation cooperative agreement OAC-1836650 (IRIS-HEP) and OAC-1450377 (DIANA/HEP). Any opinions, findings, conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

[gitter-badge]:            https://badges.gitter.im/HSF/PyHEP-histogramming.svg
[gitter-link]:             https://gitter.im/HSF/PyHEP-histogramming?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
[azure-badge]:             https://dev.azure.com/scikit-hep/boost-histogram/_apis/build/status/bh-tests?branchName=develop
[azure-link]:              https://dev.azure.com/scikit-hep/boost-histogram/_build/latest?definitionId=2&branchName=develop
[rtd-badge]:               https://readthedocs.org/projects/boost-histogram/badge/?version=latest
[rtd-link]:                https://boost-histogram.readthedocs.io/en/latest/?badge=latest
[actions-badge]:           https://github.com/scikit-hep/boost-histogram/workflows/Tests/badge.svg
[actions-link]:            https://github.com/scikit-hep/boost-histogram/actions

[Boost::Histogram]:        https://www.boost.org/doc/libs/1_71_0/libs/histogram/doc/html/index.html
[Boost::Histogram source]: https://github.com/boostorg/histogram
[fastest libraries]:       https://iscinumpy.gitlab.io/post/histogram-speeds-in-python/
[conda-link]:              https://github.com/conda-forge/boost-histogram-feedstock
