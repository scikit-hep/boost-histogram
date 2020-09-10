<img alt="boost-histogram logo" width="402" src="https://raw.githubusercontent.com/scikit-hep/boost-histogram/develop/docs/_images/BoostHistogramPythonLogo.png"/>

# boost-histogram for Python

[![Gitter][gitter-badge]][gitter-link]
[![Stack Overflow][stack-overflow-badge]][stack-overflow-link]
[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![DOI](https://zenodo.org/badge/148885351.svg)](https://zenodo.org/badge/latestdoi/148885351)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![PyPI version](https://badge.fury.io/py/boost-histogram.svg)](https://pypi.org/project/boost-histogram/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/boost-histogram)][conda-link]
[![Scikit-HEP][sk-badge]](https://scikit-hep.org/)
[![Travis-CI][travis-badge]][travis-link]

Python bindings for [Boost::Histogram][] ([source][Boost::Histogram source]), a
C++14 library. This is of the [fastest libraries][] for
histogramming, while still providing the power of a full histogram object. See [what's new](./docs/CHANGELOG.md).

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
hist = bh.Histogram(
    bh.axis.Regular(2, 0, 1),
    bh.axis.Regular(4, 0.0, 1.0),
)

# Filling can be done with arrays, one per dimension
hist.fill(
    [0.3, 0.5, 0.2], [0.1, 0.4, 0.9]
)

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
    * `bh.axis.Boolean()`: A True/False axis [(known issue with slicing/selection in 0.8.0)]()
* Axis features:
    * `.index(value)`: The index at a point (or points) on the axis
    * `.value(index)`: The value for a fractional bin (or bins) in the axis
    * `.bin(i)`: The bin edges (continuous axis) or a bin value (discrete axis)
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
  * `h.ndim`: The number of dimensions
  * `h.size or len(h)`: The number of bins
  * `+`: Add two histograms (storages must match types currently)
  * `*=`: Multiply by a scaler (not all storages) (`hist * scalar` and `scalar * hist` supported too)
  * `/=`: Divide by a scaler (not all storages) (`hist / scalar` supported too)
  * `.sum(flow=False)`: The total count of all bins
  * `.project(ax1, ax2, ...)`: Project down to listed axis (numbers)
  * `.to_numpy(flow=False)`: Convert to a Numpy style tuple (with or without under/overflow bins)
  * `.view(flow=False)`: Get a view on the bin contents (with or without under/overflow bins)
  * `.reset()`: Set counters to 0
  * `.empty(flow=False)`: Check to see if the histogram is empty (can check flow bins too if asked)
  * `.copy(deep=False)`: Make a copy of a histogram
  * `.axes`: Get the axes as a tuple-like (all properties of axes are available too)
      * `.axes[0]`: Get the 0th axis
      * `.axes.edges`: The lower values as a broadcasting-ready array
      * `.axes.centers`: The centers of the bins broadcasting-ready array
      * `.axes.widths`: The bin widths as a broadcasting-ready array
      * `.axes.metadata`: A tuple of the axes metadata
      * `.axes.size`: A tuple of the axes sizes (size without flow)
      * `.axes.extent`: A tuple of the axes extents (size with flow)
      * `.axes.bin(*args)`: Returns the bin edges as a tuple of pairs (continuous axis) or values (describe)
      * `.axes.index(*args)`: Returns the bin index at a value for each axis
      * `.axes.value(*args)`: Returns the bin value at an index for each axis
* Indexing - Supports the [Unified Histogram Indexing (UHI)](https://boost-histogram.readthedocs.io/en/latest/usage/indexing.html) proposal
    * Bin content access / setting
        * `v = h[b]`: Access bin content by index number
        * `v = h[{0:b}]`: All actions can be represented by `axis:item` dictionary instead of by position (mostly useful for slicing)
    * Slicing to get histogram or set array of values
        * `h2 = h[a:b]`: Access a slice of a histogram, cut portions go to flow bins if present
        * `h2 = h[:, ...]`: Using `:` and `...` supported just like Numpy
        * `h2 = h[::sum]`: Third item in slice is the "action"
        * `h[...] = array`: Set the bin contents, either include or omit flow bins
    * Special accessors
        * `bh.loc(v)`: Supply value in axis coordinates instead of bin number
        * `bh.underflow`: The underflow bin (use empty beginning on slice for slicing instead)
        * `bh.overflow`: The overflow bin (use empty end on slice for slicing instead)
    * Special actions (third item in slice)
        * `sum`: Remove axes via projection; if limits are given, use those
        * `bh.rebin(n)`: Rebin an axis
* NumPy compatibility
    * `bh.numpy` provides faster [drop in replacements](https://boost-histogram.readthedocs.io/en/latest/usage/numpy.html) for NumPy histogram functions
    * Histograms follow the buffer interface, and provide `.view()`
    * Histograms can be converted to NumPy style output tuple with `.to_numpy()`
* Details
    * Use `bh.Histogram(..., storage=...)` to make a histogram (there are several different types)
    * All objects support copy/deepcopy/pickle


## Supported platforms

#### Binaries available:

The easiest way to get boost-histogram is to use a binary wheel, which happens when you run:

```bash
python -m pip install boost-histogram
```

These are the supported platforms for which wheels are produced using [cibuildwheel](https://cibuildwheel.readthedocs.io/en/stable/):

| System | Arch | Python versions |
|---------|-----|------------------|
| ManyLinux1 (custom GCC 9.2) | 32 & 64-bit | 2.7, 3.5, 3.6, 3.7, 3.8 |
| ManyLinux2010 | 32 & 64-bit | 2.7, 3.5, 3.6, 3.7, 3.8 |
| macOS 10.9+ | 64-bit | 2.7, 3.5, 3.6, 3.7, 3.8 |
| Windows | 32 & 64-bit | 2.7, 3.5, 3.6, 3.7, 3.8 |


* manylinux1: Using a custom docker container with GCC 9.2; should work but can't be called directly other compiled extensions unless they do the same thing (think that's the main caveat). Supporting 32 bits because it's there.
* manylinux2010: Requires pip 10+ and a version of Linux newer than 2010.
* Windows: pybind11 requires compilation with a newer copy of Visual Studio than Python 2.7's Visual Studio 2008; you need to have the [Visual Studio 2015 distributable][msvc2015] installed (the dll is included in 2017 and 2019, as well).

[msvc2015]: https://www.microsoft.com/en-us/download/details.aspx?id=48145

If you are on a Linux system that is not part of the "many" in manylinux, such as Alpine or ClearLinux, building from source is usually fine, since the compilers on those systems are often quite new. It will just take a little longer to install when it's using the sdist instead of a wheel.


#### Conda-Forge

The boost-histogram package is available on Conda-Forge, as well. All supported versions are available with the exception of Python 2.7, which is no longer supported by conda-forge directly. If you really need boost-histogram + Conda + Python 2.7, please open an issue.

```bash
conda install -c conda-forge boost-histogram
```

#### Source builds

For a source build, for example from an "sdist" package, the only requirements are a C++14 compatible compiler. The compiler requirements are dictated by Boost.Histogram's C++ requirements: gcc >= 5.5, clang >= 3.8, msvc >= 14.1. You should have a version of pip less than 2-3 years old (10+).

If you are using Python 2.7 on Windows, you will need to use a recent version of Visual studio and force distutils to use it, or just upgrade to Python 3.6 or newer. Check the pybind11 documentation for [more help](https://pybind11.readthedocs.io/en/stable/faq.html#working-with-ancient-visual-studio-2009-builds-on-windows). On some Linux systems, you may need to use a newer compiler than the one your distribution ships with.

Numpy is downloaded during the build (enables multithreaded builds). Boost is not required or needed (this only depends on included header-only dependencies).This library is under active development; you can install directly from GitHub if you would like.

```bash
python -m pip install git+https://github.com/scikit-hep/boost-histogram.git@develop
```

## Developing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for details on how to set up a development environment.

## Contributors

We would like to acknowledge the contributors that made this project possible ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="http://iscinumpy.gitlab.io"><img src="https://avatars1.githubusercontent.com/u/4616906?v=4" width="100px;" alt=""/><br /><sub><b>Henry Schreiner</b></sub></a><br /><a href="#maintenance-henryiii" title="Maintenance">🚧</a> <a href="https://github.com/scikit-hep/boost-histogram/commits?author=henryiii" title="Code">💻</a> <a href="https://github.com/scikit-hep/boost-histogram/commits?author=henryiii" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/HDembinski"><img src="https://avatars0.githubusercontent.com/u/2631586?v=4" width="100px;" alt=""/><br /><sub><b>Hans Dembinski</b></sub></a><br /><a href="#maintenance-HDembinski" title="Maintenance">🚧</a> <a href="https://github.com/scikit-hep/boost-histogram/commits?author=HDembinski" title="Code">💻</a></td>
    <td align="center"><a href="http://lovelybuggies.github.io"><img src="https://avatars3.githubusercontent.com/u/29083689?v=4" width="100px;" alt=""/><br /><sub><b>N!no</b></sub></a><br /><a href="https://github.com/scikit-hep/boost-histogram/commits?author=LovelyBuggies" title="Tests">⚠️</a> <a href="https://github.com/scikit-hep/boost-histogram/commits?author=LovelyBuggies" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/jpivarski"><img src="https://avatars0.githubusercontent.com/u/1852447?v=4" width="100px;" alt=""/><br /><sub><b>Jim Pivarski</b></sub></a><br /><a href="#ideas-jpivarski" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/nsmith-"><img src="https://avatars3.githubusercontent.com/u/6587412?v=4" width="100px;" alt=""/><br /><sub><b>Nicholas Smith</b></sub></a><br /><a href="https://github.com/scikit-hep/boost-histogram/issues?q=author%3Ansmith-" title="Bug reports">🐛</a></td>
    <td align="center"><a href="http://www-pnp.physics.ox.ac.uk/~huffman/"><img src="https://avatars3.githubusercontent.com/u/45843291?v=4" width="100px;" alt=""/><br /><sub><b>physicscitizen</b></sub></a><br /><a href="https://github.com/scikit-hep/boost-histogram/issues?q=author%3Aphysicscitizen" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/chanchal-kumar-maji-9230a9145/"><img src="https://avatars1.githubusercontent.com/u/31502077?v=4" width="100px;" alt=""/><br /><sub><b>Chanchal Kumar Maji</b></sub></a><br /><a href="https://github.com/scikit-hep/boost-histogram/commits?author=ChanchalKumarMaji" title="Documentation">📖</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://ddavis.io/"><img src="https://avatars2.githubusercontent.com/u/3202090?v=4" width="100px;" alt=""/><br /><sub><b>Doug Davis</b></sub></a><br /><a href="https://github.com/scikit-hep/boost-histogram/issues?q=author%3Adouglasdavis" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/pgrimaud"><img src="https://avatars1.githubusercontent.com/u/1866496?v=4" width="100px;" alt=""/><br /><sub><b>Pierre Grimaud</b></sub></a><br /><a href="https://github.com/scikit-hep/boost-histogram/commits?author=pgrimaud" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/beojan"><img src="https://avatars0.githubusercontent.com/u/3727925?v=4" width="100px;" alt=""/><br /><sub><b>Beojan Stanislaus</b></sub></a><br /><a href="https://github.com/scikit-hep/boost-histogram/issues?q=author%3Abeojan" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/Popinaodude"><img src="https://avatars2.githubusercontent.com/u/20911987?v=4" width="100px;" alt=""/><br /><sub><b>Popinaodude</b></sub></a><br /><a href="https://github.com/scikit-hep/boost-histogram/issues?q=author%3APopinaodude" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/colizz"><img src="https://avatars2.githubusercontent.com/u/44885400?v=4" width="100px;" alt=""/><br /><sub><b>Congqiao Li</b></sub></a><br /><a href="https://github.com/scikit-hep/boost-histogram/issues?q=author%3Acolizz" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/alexander-held"><img src="https://avatars0.githubusercontent.com/u/45009355?v=4" width="100px;" alt=""/><br /><sub><b>alexander-held</b></sub></a><br /><a href="https://github.com/scikit-hep/boost-histogram/issues?q=author%3Aalexander-held" title="Bug reports">🐛</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification.

## Talks and other documentation/tutorial sources

The [official documentation is here](https://boost-histogram.readthedocs.io/en/latest/index.html), and includes a [quickstart](https://boost-histogram.readthedocs.io/en/latest/usage/quickstart.html).


* [2019-4-15 IRIS-HEP Topical meeting](https://indico.cern.ch/event/803122/)
* [2019-10-17 PyHEP Histogram session](https://indico.cern.ch/event/833895/contributions/3577833/) - [repo with talks and workbook](https://github.com/henryiii/pres-bhandhist)
* [2019-11-7 CHEP](https://indico.cern.ch/event/773049/contributions/3473265/)
* [2020-07-07 SciPy](https://www.youtube.com/watch?v=ERraTfHkPd0&list=PLYx7XA2nY5GfY4WWJjG5cQZDc7DIUmn6Z&index=4)
* [2020-07-17 PyHEP](https://indico.cern.ch/event/882824/contributions/3931299/)

---

## Acknowledgements

This library was primarily developed by Henry Schreiner and Hans Dembinski.

Support for this work was provided by the National Science Foundation cooperative agreement OAC-1836650 (IRIS-HEP) and OAC-1450377 (DIANA/HEP). Any opinions, findings, conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

[gitter-badge]:            https://badges.gitter.im/HSF/PyHEP-histogramming.svg
[gitter-link]:             https://gitter.im/HSF/PyHEP-histogramming?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
[rtd-badge]:               https://readthedocs.org/projects/boost-histogram/badge/?version=latest
[rtd-link]:                https://boost-histogram.readthedocs.io/en/latest/?badge=latest
[actions-badge]:           https://github.com/scikit-hep/boost-histogram/workflows/Tests/badge.svg
[actions-link]:            https://github.com/scikit-hep/boost-histogram/actions
[travis-link]:             https://travis-ci.com/scikit-hep/boost-histogram
[travis-badge]:            https://travis-ci.com/scikit-hep/boost-histogram.svg?branch=master

[Boost::Histogram]:        https://www.boost.org/doc/libs/release/libs/histogram/doc/html/index.html
[Boost::Histogram source]: https://github.com/boostorg/histogram
[fastest libraries]:       https://iscinumpy.gitlab.io/post/histogram-speeds-in-python/
[conda-link]:              https://github.com/conda-forge/boost-histogram-feedstock
[sk-badge]:                https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
[stack-overflow-badge]:    https://img.shields.io/badge/stack-overflow-orange
[stack-overflow-link]:     https://stackoverflow.com/questions/tagged/boost-histogram
