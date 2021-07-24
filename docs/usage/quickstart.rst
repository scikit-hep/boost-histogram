.. _usage-quickstart:

Quickstart
==========

All of the examples will assume the following import:

.. code:: python3

   import boost_histogram as bh

In boost-histogram, a histogram is collection of Axis objects and a
storage.


.. image:: ../_images/histogram_design.png
   :alt: Regular axis illustration
   :align: center

Making a histogram
------------------

You can make a histogram like this:

.. code:: python3

   hist = bh.Histogram(bh.axis.Regular(bins=10, start=0, stop=1))

If you’d like to type less, you can leave out the keywords:

.. code:: python3

   hist = bh.Histogram(bh.axis.Regular(10, 0, 1))


The exact same syntax is used for 1D, 2D, and ND histograms:

.. code:: python3

   hist3D = bh.Histogram(
       bh.axis.Regular(10, 0, 100, circular=True),
       bh.axis.Regular(10, 0.0, 10.0),
       bh.axis.Variable([1, 2, 3, 4, 5, 5.5, 6]),
   )

See :ref:`usage-axes` and :ref:`usage-transforms`.

You can also select a different storage with the ``storage=`` keyword argument;
see :ref:`usage-storage` for details about the other storages.

Filling a histogram
-------------------

Once you have a histogram, you can fill it using ``.fill``. Ideally, you
should give arrays, but single values work as well:

.. code:: python3

   hist = bh.Histogram(bh.axis.Regular(10, 0.0, 1.0))
   hist.fill(0.9)
   hist.fill([0.9, 0.3, 0.4])


Slicing and rebinning
---------------------

You can slice into a histogram using bin coordinates or data coordinates using
``bh.loc(v)``. You can also rebin with ``bh.rebin(n)`` or remove an entire axis
using ``sum`` (technically as the third slice argument, though it is allowed by
itself as well):

.. code:: python3

    hist = bh.Histogram(
        bh.axis.Regular(10, 0, 1),
        bh.axis.Regular(10, 0, 1),
        bh.axis.Regular(10, 0, 1),
    )
    mini = hist[1:5, bh.loc(0.2) : bh.loc(0.9), sum]
    # Will be 4 bins x 7 bins

See :ref:`usage-indexing`.

Accessing the contents
----------------------

You can use ``hist.values()`` to get a NumPy array from any histogram. You can
get the variances with ``hist.variances()``, though if you fill an unweighted
storage with weights, this will return None, as you no longer can compute the
variances correctly (please use a weighted storage if you need to). You can
also get the number of entries in a bin with ``.counts()``; this will return
counts even if your storage is a mean storage. See :ref:`usage-plotting`.

If you want access to the full underlying storage, ``.view()`` will return a
NumPy array for simple storages or a RecArray-like wrapper for non-simple
storages.  Most methods  offer an optional keyword argument that you can pass,
``flow=True``, to enable the under and overflow bins (disabled by default).

.. code:: python3

    np_array = hist.view()


Setting the contents
--------------------

You can set the contents directly as you would a NumPy array;
you can set either values or arrays at a time:

.. code:: python3

    hist[2] = 3.5
    hist[bh.underflow] = 0  # set the underflow bin
    hist2d[3:5, 2:4] = np.eye(2)  # set with array

For non-simple storages, you can add an extra dimension that matches the
constructor arguments of that accumulator. For example, if you want to fill
a Weight histogram with three values, you can dimension:

.. code:: python3

    hist[0:3] = [[1, 0.1], [2, 0.2], [3, 0.3]]

See :ref:`usage-indexing`.

Accessing Axes
--------------

The axes are directly available in the histogram, and you can access
a variety of properties, such as the ``edges`` or the ``centers``. All
properties and methods are also available directly on the ``axes`` tuple:

.. code:: python3

   ax0 = hist.axes[0]
   X, Y = hist.axes.centers

See :ref:`usage-axes`.


Saving Histograms
-----------------

You can save histograms using pickle:

.. code:: python3

    import pickle

    with open("file.pkl", "wb") as f:
        pickle.dump(h, f)

    with open("file.pkl", "rb") as f:
        h2 = pickle.load(f)

    assert h == h2

Special care was taken to ensure that this is fast and efficient.  Please use
the latest version of the Pickle protocol you feel comfortable using; you
cannot use version 0, the version that was default on Python 2. The most recent
versions provide performance benefits.

Computing with Histograms
-------------------------

As an complete example, let's say you wanted to compute and plot the density:

.. literalinclude:: ../../examples/simple_density.py
   :language: python

.. image:: ../_images/ex_hist_density.png
   :alt: Density histogram output
   :align: center

Comparing with Boost.Histogram
------------------------------

This is built on the Boost.Histogram library.

See :ref:`usage-comparison`.
