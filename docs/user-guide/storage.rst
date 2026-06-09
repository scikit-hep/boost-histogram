.. _usage-storage:

Storages
========

There are several storages to choose from. To select a storage, pass the
``storage=bh.storage.`` argument when making a histogram.

Simple storages
---------------

These storages hold a single value that keeps track of a count, possibly a
weighted count.

Double
^^^^^^

By default, boost-histogram selects the ``Double()`` storage. For most uses,
this should be ideal. It is just as fast as the ``Int64()`` storage, it can fill
up to 53 bits of information (9 quadrillion) counts per cell, and supports
weighted fills. It can also be scaled by a floating point values without making
a copy.

.. code-block:: python3

    h = bh.Histogram(bh.axis.Regular(10, 0, 1))  # Double() is the default
    h.fill([0.2, 0.3], weight=[0.5, 2])  # Weights are optional
    print(f"{h[bh.loc(.2)]=}\n{h[bh.loc(.3)]=}")

.. code-block:: text

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
non-integer fills for data that should represent raw, unweighted counts.

.. code-block:: python3

    h = bh.Histogram(bh.axis.Regular(10, 0, 1), storage=bh.storage.Int64())
    h.fill([0.2, 0.3], weight=[1, 2])  # Integer weights supported
    print(f"{h[bh.loc(.2)]=}\n{h[bh.loc(.2)]=}")

.. code-block:: text

    h[bh.loc(.2)]=1
    h[bh.loc(.3)]=2


AtomicInt64
^^^^^^^^^^^

This storage is like ``Int64()``, but also provides a thread safety guarantee.
You can fill a single histogram from multiple threads.


Sparse storages
---------------

DoubleSparse
^^^^^^^^^^^^

The ``DoubleSparse()`` storage is like ``Double()``, but only the bins that are
actually filled are stored (in a hash map keyed on the bin index). This makes
histograms over very large or high-dimensional axis spaces feasible, where a
dense grid would not fit in memory even though only a small fraction of the bins
are ever filled.

.. code-block:: python3

    import boost_histogram as bh

    # A 5-dimensional grid; dense storage would need 10**15 cells.
    axes = [bh.axis.Regular(1000, 0, 1) for _ in range(5)]
    h = bh.Histogram(*axes, storage=bh.storage.DoubleSparse())
    h.fill(*data)  # only the filled cells are allocated

Because the data is not stored as a dense array, **sparse storage is not a
drop-in replacement for a dense histogram**:

* ``h.view()`` and ``np.asarray(h)`` raise, since there is no contiguous buffer
  to view into.
* The copying accessors -- ``h.values()``, ``h.variances()``, ``h.counts()`` --
  *densify* into a freshly allocated array, so they keep working. This is only a
  good idea when the dense shape actually fits in memory; for a genuinely huge
  grid, use :meth:`~boost_histogram.Histogram.to_coo` instead.
* Reading a single bin (``h[bh.loc(...)]``), slicing, projection, ``.sum()``, and
  adding two histograms together all keep the result sparse.
* Writes are fill-only: ``h[...] = value`` is not supported (it would need a live
  buffer).

Only ``double`` sparse storage is available for now; accumulator-backed sparse
storages (the equivalent of ``Weight``, ``Mean``, ...) are waiting on a
Boost.Histogram fix that ships in Boost 1.92.

Reading the filled cells
""""""""""""""""""""""""

:meth:`~boost_histogram.Histogram.to_coo` returns the filled cells in coordinate
(COO) form -- a tuple of per-axis index arrays (in the style of
:func:`numpy.nonzero`, so they can be used directly as a fancy index) and the
array of corresponding values:

.. code-block:: python3

    h = bh.Histogram(bh.axis.Regular(5, 0, 5), storage=bh.storage.DoubleSparse())
    h.fill([1.5, 1.5, 3.5])

    indices, values = h.to_coo()
    # indices == (array([1, 3]),)   values == array([2., 1.])

Pass ``flow=True`` to include the underflow/overflow bins (the indices then run
over the with-flow grid, with underflow at ``0``).

Converting to a SciPy sparse array
""""""""""""""""""""""""""""""""""

For a 2D histogram, the COO output maps directly onto a
:class:`scipy.sparse.coo_array`. The per-axis index arrays are the row and
column coordinates, and ``h.axes.size`` is the dense shape:

.. code-block:: python3

    from scipy.sparse import coo_array

    h = bh.Histogram(
        bh.axis.Regular(1000, 0, 1000),
        bh.axis.Regular(1000, 0, 1000),
        storage=bh.storage.DoubleSparse(),
    )
    h.fill(xs, ys)

    (rows, cols), values = h.to_coo()
    matrix = coo_array((values, (rows, cols)), shape=h.axes.size)

(SciPy sparse arrays are two-dimensional, so this applies to 2D histograms; for
higher dimensions, use the ``(indices, values)`` pair directly, e.g. with an
N-dimensional sparse library such as ``sparse``.)


Accumulator storages
--------------------

These storages hold more than one number internally. They return a smart view when queried
with ``.view()``; see :ref:`usage-accumulators` for information on each accumulator and view.

Weight
^^^^^^

This storage keeps a sum of weights as well (in CERN ROOT, this is like calling
``.Sumw2()`` before filling a histogram). It uses the ``WeightedSum`` accumulator.

MultiCell
^^^^^^^^^

This storage is like the ``Double`` storage but supports storing multiple entries per bin independently. This is useful if one has to deal with many independent weights per event that all correspond to the same parameter (they will a be binned into the same bin) and have to be summed independently (e.g. to track the effect of systematic variations). It is supposed to be filled much faster compared to filling many histograms with a ``Weight`` storage type in a loop if one deals with a large number of different weights.

The number of entries per bin has to be fixed and is provided to the storage at its construction through

.. code-block:: python3

    bh.Histogram(…, storage=MultiCell(nelem))

where ``nelem`` is the number of entries per bin.
The entries have to provided as a 2-dimensional array ``(n, nelem)`` with the first dimension being the events to histogram and the second axis being the entries per event.
To fill a histogram ``h`` one has to provide the entries via the ``weight`` keyword:

.. code-block:: python3

    h.fill(..., weight=weights)

Any slicing or projection operation works for ``MultiCell`` histograms identical to any other histogram with different storage type, the entries are here not considered an additional axis for the histogram.
Calling ``h.view()`` returns an array where the entries are indexed as the first axis (e.g. ``h.view()[0]`` is the histogram content for the first entry per bin).
Contrary to the ``Weight`` storage the ``MultiCell`` storage does not track variances (it does not track the sum of weights squared) because this might not be necessary for every entry index. Instead, the user is supposed to track the variances themselves if required. This could be achieved by providing the variances as a separate entry to the ``MultiCell`` histogram by increasing the number of entries that are stored per bin (e.g. one could provide the square of ``weights[:, 0]`` as ``weights[:, 1]`` to track the sum of weight squared of the first entry index in the second entry index).
Note: If you should ever need to use the lowlevel ``h._hist.fill()`` function with a ``MultiCell`` storage you will have to use the ``sample`` keyword to pass the weights instead of the ``weight`` keyword because that is used on the C++ side, but the highlevel python boost_histogram API hides this from the user.


Mean
^^^^

This storage tracks a "Profile", that is, the mean value of the accumulation instead of the sum.
It stores the count (as a double), the mean, and a term that is used to compute the variance. When
filling, you must add a ``sample=`` term.


WeightedMean
^^^^^^^^^^^^

This is similar to Mean, but also keeps track a sum of weights like term as well.
