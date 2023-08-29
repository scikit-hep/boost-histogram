.. _usage-numpy:

NumPy compatibility
===================

Histogram conversion
--------------------

Accessing the storage array
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can access the storage of any Histogram using ``.view()``, see
:ref:`usage-histogram`.

NumPy tuple output
^^^^^^^^^^^^^^^^^^

You can directly convert a histogram into the tuple of outputs that
``np.histogram*`` would give you using ``.to_numpy()`` or
``.to_numpy(flow=True)`` on any histogram.  This returns
``edges[0], edges[1], ..., values``, and the edges are NumPy-style (upper edge
inclusive).

NumPy adaptors
--------------

You can use boost-histogram as a drop in replacement for NumPy histograms.  All
three histogram functions (``bh.numpy.histogram``, ``bh.numpy.histgoram2d``, and
``bh.numpy.histogramdd``) are provided. The syntax is identical, though
boost-histogram adds three new keyword-only arguments; ``storage=`` to select the
storage, ``histogram=bh.Histogram`` to produce a boost-histogram instead of a
tuple, and ``threads=N`` to select a number of threads to fill with.

1D histogram example
^^^^^^^^^^^^^^^^^^^^

If you try the following in an IPython session, you will get:

.. code-block:: python3

   import numpy as np
   import boost_histogram as bh

   norm_vals = np.concatenate(
       [
           np.random.normal(loc=5, scale=1, size=1_000_000),
           np.random.normal(loc=2, scale=0.2, size=200_000),
           np.random.normal(loc=8, scale=0.2, size=200_000),
       ]
   )

.. code-block:: ipython

   %%timeit
   bins, edges = np.histogram(norm_vals, bins=100, range=(0, 10))

.. code-block:: text

   17.4 ms ± 2.64 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)

Of course, you then are either left on your own to compute centers,
density, widths, and more, or in some cases you can change the
computation call itself to add ``density=``, or use the matching
function inside Matplotlib, and the API is different if you want 2D or
ND histograms. But if you already use NumPy histograms and you really
don’t want to rewrite your code, boost-histogram has adaptors for the
three histogram functions in NumPy:

.. code-block:: ipython

   %%timeit
   bins, edges = bh.numpy.histogram(norm_vals, bins=100, range=(0, 10))

.. code-block:: text

   7.3 ms ± 55.7 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

This is only a hair slower than using the raw boost-histogram API,
and is still a nice performance boost over NumPy. You can even use the
NumPy syntax if you want a boost-histogram object later:

.. code-block:: python3

   hist = bh.numpy.histogram(norm_vals, bins=100, range=(0, 10), histogram=bh.Histogram)

You can later get a NumPy style output tuple from a histogram object:

.. code-block:: python3

   bins, edges = hist.to_numpy()

So you can transition your code slowly to boost-histogram.


2D Histogram example
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python3

   data = np.random.multivariate_normal((0, 0), ((1, 0), (0, 0.5)), 10_000_000).T.copy()

We can check the performance against NumPy again; NumPy does not do well
with regular spaced bins in more than 1D:

.. code-block:: ipython

   %%timeit
   np.histogram2d(*data, bins=(400, 200), range=((-2, 2), (-1, 1)))

.. code-block:: text

   1.31 s ± 17.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

.. code-block:: ipython

   %%timeit
   bh.numpy.histogram2d(*data, bins=(400, 200), range=((-2, 2), (-1, 1)))

.. code-block:: text

   101 ms ± 117 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

For more than one dimension, boost-histogram is more than an order of
magnitude faster than NumPy for regular spaced binning. Although
optimizations may be added to boost-histogram for common axes
combinations later, in 0.6.1, all axes combinations share a common code
base, so you can expect *at least* this level of performance regardless
of the axes types or number of axes! Threaded filling can give you an
even larger performance boost if you have multiple cores and a large
fill to perform.
