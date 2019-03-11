# boost-histogram for Python

[![Gitter](https://badges.gitter.im/HSF/PyHEP-histogramming.svg)](https://gitter.im/HSF/PyHEP-histogramming?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

Python bindings for [Boost::Histogram][], a C++14 library. This should become one of the [fastest libraries][] for histogramming, while still providing the power of a full histogram object.

> # Warning: This bindings are in progress and are not yet in an alpha stage.
>
> Join the discussion on gitter to follow the development!

[Boost::Histogram]:  https://www.boost.org/doc/libs/develop/libs/histogram/doc/html/index.html 
[fastest libraries]: https://iscinumpy.gitlab.io/post/histogram-speeds-in-python/

## Hacking

This repository has dependencies in submodules. Check out the repository like this.

    git clone https://github.com/scikit-hep/boost-histogram.git
    cd boost-histogram
    git submodule update --init --depth 10

Make a build directory and run cmake.

    mkdir build
    cd build
    cmake .. 
    make -j4

Run the unit tests.

    make test
