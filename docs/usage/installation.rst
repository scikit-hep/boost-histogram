.. _usage-installation:

Installation
============


Boost-histogram (`source <https://github.com/scikit-hep/boost-histogram>`__) is a Python package providing Python bindings for Boost.Histogram_ (`source <https://github.com/boostorg/histogram>`__).

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

These are the supported platforms for which wheels are produced:

=========================== =========== =======================
System                      Arch        Python versions
=========================== =========== =======================
ManyLinux1 (custom GCC 9.2) 64 & 32-bit 2.7, 3.5, 3.6, 3.7, 3.8
ManyLinux2010               64-bit      2.7, 3.5, 3.6, 3.7, 3.8
macOS 10.9+                 64-bit      2.7, 3.5, 3.6, 3.7, 3.8
Windows                     64 & 32-bit 2.7, 3.5, 3.6, 3.7, 3.8
=========================== =========== =======================

-  manylinux1: Using a custom docker container with GCC 9.2; should work
   but can't be called directly other compiled extensions unless they do
   the same thing (think that’s the main caveat). Supporting 32 bits
   because it’s there.
-  manylinux2010: Requires pip 10+ and a version of Linux newer than
   2010.
-  Windows: pybind11 requires compilation with a newer copy of Visual
   Studio than Python 2.7’s Visual Studio 2008; you need to have the
   `Visual Studio 2015
   distributable <https://www.microsoft.com/en-us/download/details.aspx?id=48145>`__
   installed (the dll is included in 2017 and 2019, as well).

If you are on a Linux system that is not part of the “many” in
manylinux, such as Alpine or ClearLinux, building from source is usually
fine, since the compilers on those systems are often quite new. It will
just take a little longer to install when it’s using the sdist instead
of a wheel.

Conda-Forge
^^^^^^^^^^^

The boost-histogram package is available on Conda-Forge, as well. All
supported versions are available with the exception of Windows + Python
2.7, which cannot built due to the age of the compiler. Please use Pip
if you *really* need Python 2.7 on Windows. You will also need the VS
2015 distributable, as described above.

::

   conda install -c conda-forge boost-histogram

Source builds
^^^^^^^^^^^^^

For a source build, for example from an “sdist” package, the only
requirements are a C++14 compatible compiler. The compiler requirements
are dictated by Boost.Histogram’s C++ requirements: gcc >= 5.5, clang >=
3.8, msvc >= 14.1. You should have a version of pip less than 2-3 years
old (10+).

If you are using Python 2.7 on Windows, you will need to use a recent
version of Visual studio and force distutils to use it, or just upgrade
to Python 3.6 or newer. Check the pybind11 documentation for `more
help <https://pybind11.readthedocs.io/en/stable/faq.html#working-with-ancient-visual-studio-2009-builds-on-windows>`__.
On some Linux systems, you may need to use a newer compiler than the one
your distribution ships with.

Numpy is downloaded during the build (enables multithreaded builds).
Boost is not required or needed (this only depends on included
header-only dependencies).This library is under active development; you
can install directly from GitHub if you would like.

.. code:: bash

   python -m pip install git+https://github.com/scikit-hep/boost-histogram.git@develop
