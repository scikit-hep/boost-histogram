.. _usage-installation:

Installation
============


Boost-histogram (`source <https://github.com/scikit-hep/boost-histogram>`__) is
a Python package providing Python bindings for Boost.Histogram_ (`source
<https://github.com/boostorg/histogram>`__).

.. _Boost.Histogram: https://www.boost.org/doc/libs/release/libs/histogram/doc/html/index.html


You can install this library from
`PyPI <https://pypi.org/project/boost-histogram/>`__ with pip:

.. code:: bash

   python -m pip install boost-histogram

or you can use Conda through
`conda-forge <https://github.com/conda-forge/boost-histogram-feedstock>`__:

.. code:: bash

   conda install -c conda-forge boost-histogram

All the normal best-practices for Python apply; you should be in a
virtual environment, etc.



Supported platforms
-------------------

Binaries available:
^^^^^^^^^^^^^^^^^^^

The supported platforms are listed in the README - All common linux
machines, all common macOS versions, and all common Windows versions.

Conda-Forge
^^^^^^^^^^^

The boost-histogram package is available on Conda-Forge, as well. All
supported versions are available.

::

   conda install -c conda-forge boost-histogram

Source builds
^^^^^^^^^^^^^

For a source build, for example from an “sdist” package, the only
requirements are a C++14 compatible compiler. The compiler requirements
are dictated by Boost.Histogram’s C++ requirements: gcc >= 5.5, clang >=
3.8, msvc >= 14.1. You should have a version of pip less than 2-3 years
old (10+).

NumPy is downloaded during the build (enables multithreaded builds).
Boost is not required or needed (this only depends on included
header-only dependencies).This library is under active development; you
can install directly from GitHub if you would like.

.. code:: bash

   python -m pip install git+https://github.com/scikit-hep/boost-histogram.git@develop
