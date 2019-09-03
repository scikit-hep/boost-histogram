Quickstart
==========

All of the examples will assume the following import:

.. code:: python

   import boost.histogram as bh

In boost-histogram, a histogram is collection of Axis objects and a
storage.


.. image:: ../_images/histogram_design.png
   :alt: Regular axis illustration
   :align: center

Making a histogram
------------------

You can make a histogram like this:

.. code:: python

   hist = bh.histogram(bh.axis.regular(bins=10, start=0, stop=1))

If you’d like to type less, you can leave out the keywords:

.. code:: python

   hist = bh.histogram(bh.axis.regular(10, 0, 1))


The exact same syntax is used for 1D, 2D, and ND histograms:

.. code:: python

   hist3D = bh.histogram(
       bh.axis.circular(10, 0, 100),
       bh.axis.regular(10, 0.0, 10.0),
       bh.axis.variable([1,2,3,4,5,5.5,6])
   )

Filling a histogram
-------------------

Once you have a histogram, you can fill it using ``.fill``. Ideally, you
can give arrays, but single values work as well:

.. code:: python

   hist = bh.histogram((10, 0.0, 1.0))
   hist.fill(0.9)
   hist.fill([0.9, 0.3, 0.4])

You can pass ``threads=N`` to split the histogram into N copies and fill
one per thread, recombining at the end - this helps you use all your
cores. ``N=0`` will fill with a number of threads based on your
available cores.

Accessing the contents
----------------------

You can directly give a histogram to anything that expects a Python
buffer, such as ``np.asarray``. You can also use ``hist.view()`` to get
a numpy array - currently limited to simple storage types, like int and double.
Most methods like ``.view()`` offer an optional keyword
argument that you can pass, ``flow=True``, to enable the under and
overflow bins (disabled by default).

A very powerful way to access the bins is with the indexed iterator. For example,
if you want to initialize the contents of every bin to a function, you could do:

.. code:: python

   def f(x, y):
       return x**2 + y**2 # Any function here

   hist = bh.histogram(bh.axis.regular(10, 0.0, 1.0),
                       bh.axis.regular(10, 0, 10),
                       storage=bh.storage.double())
   
   for ind in hist.indexed():
       ind.content = f(*ind.centers())

