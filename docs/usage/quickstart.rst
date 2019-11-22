Quickstart
==========

All of the examples will assume the following import:

.. code:: python

   import boost_histogram as bh

In boost-histogram, a histogram is collection of Axis objects and a
storage.


.. image:: ../_images/histogram_design.png
   :alt: Regular axis illustration
   :align: center

Making a histogram
------------------

You can make a histogram like this:

.. code:: python

   hist = bh.Histogram(bh.axis.Regular(bins=10, start=0, stop=1))

If you’d like to type less, you can leave out the keywords:

.. code:: python

   hist = bh.Histogram(bh.axis.Regular(10, 0, 1))


The exact same syntax is used for 1D, 2D, and ND histograms:

.. code:: python

   hist3D = bh.Histogram(
       bh.axis.Regular(10, 0, 100, circular=True),
       bh.axis.Regular(10, 0.0, 10.0),
       bh.axis.Variable([1,2,3,4,5,5.5,6])
   )

Filling a histogram
-------------------

Once you have a histogram, you can fill it using ``.fill``. Ideally, you
can give arrays, but single values work as well:

.. code:: python

   hist = bh.Histogram(bh.axis.Regular(10, 0.0, 1.0))
   hist.fill(0.9)
   hist.fill([0.9, 0.3, 0.4])


Accessing the contents
----------------------

You can use ``hist.view()`` to get
a numpy array (or a RecArray-like wrapper for non-simple storages).
Most methods like ``.view()`` offer an optional keyword
argument that you can pass, ``flow=True``, to enable the under and
overflow bins (disabled by default).


