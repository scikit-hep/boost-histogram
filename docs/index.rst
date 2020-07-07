.. boost-histogram documentation master file, created by
   sphinx-quickstart on Tue Apr 23 12:12:27 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _images/BoostHistogramPythonLogo.png
  :width: 70%
  :alt: Boost histogram logo
  :align: center

Welcome to boost-histogram's documentation!
===========================================

|Gitter| |Build Status| |Actions Status| |Documentation Status| |DOI|
|Code style: black| |PyPI version| |Conda-Forge| |Scikit-HEP|

Boost-histogram (`source <https://github.com/scikit-hep/boost-histogram>`__) is
a Python package providing Python bindings for Boost.Histogram_ (`source
<https://github.com/boostorg/histogram>`__).  You can install this library from
`PyPI <https://pypi.org/project/boost-histogram/>`__ with pip or you can use
Conda via `conda-forge <https://github.com/conda-forge/boost-histogram-feedstock>`__:

.. code:: bash

   python -m pip install boost-histogram

.. code:: bash

   conda install -c conda-forge boost-histogram

All the normal best-practices for Python apply; you should be in a
virtual environment, etc. See :ref:`usage-installation` for more details. An example of usage:

.. code:: python3

   import boost_histogram as bh

   # Compose axis however you like; this is a 2D histogram
   hist = bh.Histogram(bh.axis.Regular(2, 0, 1),
                       bh.axis.Regular(4, 0.0, 1.0))

   # Filling can be done with arrays, one per dimension
   hist.fill([.3, .5, .2],
             [.1, .4, .9])

   # Numpy array view into histogram counts, no overflow bins
   counts = hist.view()

See :ref:`usage-quickstart` for more.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   usage/installation
   usage/quickstart
   CHANGELOG
   usage/histogram
   usage/axes
   usage/storage
   usage/accumulators
   usage/transforms
   usage/indexing
   usage/analyses
   usage/numpy
   usage/comparison


.. toctree::
  :maxdepth: 1
  :caption: Examples:

  notebooks/SimpleExample
  notebooks/aghast
  notebooks/ThreadedFills
  notebooks/PerformanceComparison
  notebooks/xarray
  notebooks/BoostHistogramHandsOn


.. toctree::
  :caption: API Reference:

  api/modules

Acknowledgements
----------------

This library was primarily developed by Henry Schreiner and Hans
Dembinski.

Support for this work was provided by the National Science Foundation
cooperative agreement OAC-1836650 (IRIS-HEP) and OAC-1450377
(DIANA/HEP). Any opinions, findings, conclusions or recommendations
expressed in this material are those of the authors and do not
necessarily reflect the views of the National Science Foundation.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Boost.Histogram: https://www.boost.org/doc/libs/release/libs/histogram/doc/html/index.html
.. |Gitter| image:: https://badges.gitter.im/HSF/PyHEP-histogramming.svg
   :target: https://gitter.im/HSF/PyHEP-histogramming?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
.. |Build Status| image:: https://dev.azure.com/scikit-hep/boost-histogram/_apis/build/status/bh-tests?branchName=develop
   :target: https://dev.azure.com/scikit-hep/boost-histogram/_build/latest?definitionId=2&branchName=develop
.. |Actions Status| image:: https://github.com/scikit-hep/boost-histogram/workflows/Tests/badge.svg
   :target: https://github.com/scikit-hep/boost-histogram/actions
.. |Documentation Status| image:: https://readthedocs.org/projects/boost-histogram/badge/?version=latest
   :target: https://boost-histogram.readthedocs.io/en/latest/?badge=latest
.. |DOI| image:: https://zenodo.org/badge/148885351.svg
   :target: https://zenodo.org/badge/latestdoi/148885351
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
.. |PyPI version| image:: https://badge.fury.io/py/boost-histogram.svg
   :target: https://pypi.org/project/boost-histogram/
.. |Conda-Forge| image:: https://img.shields.io/conda/vn/conda-forge/boost-histogram
   :target: https://github.com/conda-forge/boost-histogram-feedstock
.. |Scikit-HEP| image:: https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
   :target: https://scikit-hep.org/
