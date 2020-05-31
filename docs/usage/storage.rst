.. _usage-storage:

Storages
========

There are several storage to choose from. To select a storage, pass the
``storage=bh.storage.`` argument when making a histogram.

Simple storages
---------------

These storages hold a single value that keeps track of a count, possibly a
weighed count.

Double
^^^^^^

By default, boost-histogram selects the ``Double()`` storage. For most uses,
this should be ideal. It is just as fast as the ``Int64()`` storage, it can fill
up to 53 bits of information (9 quadrillion) counts per cell, and supports
weighted fills. It can also be scaled by a floating point values without making
a copy.

.. code:: python3

    h = bh.Histogram(bh.axis.Regular(10, 0, 1))    # Double() is the default
    h.fill([.2, .3], weight=[.5, 2])             # Weights are optional
    print(f"{h[bh.loc(.2)]=}\n{h[bh.loc(.3)]=}") # Python 3.8 print

.. code:: text

    h[bh.loc(.2)]=0.5
    h[bh.loc(.3)]=2.0

Unlimited
^^^^^^^^^

The Unlimited storage starts as an 8-bit integer and grows, and converts to a
double if weights are used (or, currently, if a view is requested). This allows
you to keep the memory usage minimal, at the expense of occasionally making an
internal copy.

Int64
^^^^^

A true integer storage is provided, as well; this storage has the ``np.uint64``
datatype.  This eventually should provide type safety by not accepting
non-integer fills for data that should represent raw, unweighed counts.

.. code:: python3

    h = bh.Histogram(bh.axis.Regular(10, 0, 1), storage=bh.storage.Int64())
    h.fill([.2, .3], weight=[1, 2])               # Integer weights supported
    print(f"{h[bh.loc(.2)]=}\n{h[bh.loc(.2)]=}")

.. code:: text

    h[bh.loc(.2)]=1
    h[bh.loc(.3)]=2


AtomicInt64
^^^^^^^^^^^

This storage is like ``Int64()``, but also provides a thread safety guarantee.
You can fill a single histogram from multiple threads.


Accumulator storages
--------------------

These storages hold more than one number internally. They return a smart view when queried
with ``.view()``; see :ref:`usage-accumulators` for information on each accumulator and view.

Weight
^^^^^^

This storage keeps a sum of weights as well (in CERN ROOT, this is like calling
``.Sumw2()`` before filling a histogram). It uses the ``WeightedSum`` accumulator.


Mean
^^^^

This storage tracks a "Profile", that is, the mean value of the accumulation instead of the sum.
It stores the count (as a double), the mean, and a term that is used to compute the variance. When
filling, you can add a ``sample=`` term.


WeightedMean
^^^^^^^^^^^^

This is similar to Mean, but also keeps track a sum of weights like term as well.
