.. _usage-accumulators:

Accumulators
============

Common properties
-----------------

All accumulators can be filled like a histogram. You just call `.fill` with
values, and this looks and behaves like filling a single-bin or "scalar"
histogram. Like histograms, the fill is inplace.

All accumulators have a `.value` property as well, which gives the primary
value being accumulated.

Types
-----

There are several accumulators.

Sum
^^^

This is the simplest accumulator, and is never returned from a histogram. This
is internally used by the Double and Unlimited storages to perform sums when
needed. It uses a highly accurate Neumaier sum to compute the floating point
sum with a correction term. Since this accumulator is never returned by a
histogram, it is not available in a view form, but only as a single accumulator
for comparison and access to the algorithm. Usage example in Python 3.8,
showing how non-accurate sums fail to produce the obvious answer, 2.0::

    import math
    import numpy as np
    import boost_histogram as bh

    values = [1.0, 1e100, 1.0, -1e100]
    print(f"{sum(values) = } (simple)")
    print(f"{math.fsum(values) = }")
    print(f"{np.sum(values) = } (pairwise)")
    print(f{bh.accumulators.Sum().fill(values) = }")

.. code:: text

    sum(values) = 0.0
    math.fsum(values) = 2.0
    np.sum(values) = 0.0
    bh.accumulators.Sum().fill(values) = Sum(2)


Note that this is still intended for performance and does not guarantee
correctness as ``math.fsum`` does. In general, you must not have more than two
orders of values::

    values = [1., 1e100, 1e50, 1., -1e50, -1e100]
    print(f"{math.fsum(values) = }")
    print(f{bh.accumulators.Sum().fill(values) = }")

.. code:: text

    math.fsum(values) = 2.0
    bh.accumulators.Sum().fill(values) = Sum(0)

You should note that this is a highly contrived example and the Sum accumulator
should still outperform simple and pairwise summation methods for a minimal
performance cost. Most notably, you have to have large cancellations with
negative values, which histograms generally do not have.

You can use ``+=`` with a float value or a Sum to fill as well.

WeightedSum
^^^^^^^^^^^

This accumulator is contained in the Weight storage, and supports Views. It
provides two values; ``.value``, and ``.variance``. The value is the sum of the
weights, and the variance is the sum of the squared weights.

For example, you could sum the following values::

    import boost_histogram as bh

    values = [10]*10
    smooth = bh.accumulators.WeightedSum().fill(values)
    print(f"{smooth = }")

    values = [1]*9 + [91]
    rough = bh.accumulators.WeightedSum().fill(values)
    print(f"{rough =  }")

.. code:: text

    smooth = WeightedSum(value=100, variance=1000)
    rough =  WeightedSum(value=100, variance=8290)

When filling, you can optionally provide a ``variance=`` keyword, with either a
single value or a matching length array of values.

You can also fill with ``+=`` on a value or another WeighedSum.

Mean
^^^^

This accumulator is contained in the Mean storage, and supports Views. It
provides three values; ``.count``, ``.value``, and ``.variance``. Internally,
the variance is stored as ``sum_of_deltas_squared``, which is used to compute
``variance``.

For example, you could compute the mean of the following values::

    import boost_histogram as bh

    values = [10]*10
    smooth = bh.accumulators.Mean().fill(values)
    print(f"{smooth = }")

    values = [1]*9 + [91]
    rough = bh.accumulators.Mean().fill(values)
    print(f"{rough =  }")

.. code:: text

    smooth = Mean(count=10, value=10, variance=0)
    rough =  Mean(count=10, value=10, variance=810)

You can add a `weight=` keyword when filling, with either a single value
or a matching length array of values.

You can call a Mean with a value or with another Mean to fill inplace, as well.

WeightedMean
^^^^^^^^^^^^

This accumulator is contained in the WeightedMean storage, and supports Views.
It provides four values; ``.sum_of_weights``, ``sum_of_weights_squared``,
``.value``, and ``.variance``. Internally, the variance is stored as
``_sum_of_weighted_deltas_squared``, which is used to compute ``variance``.

For example, you could compute the mean of the following values::

    import boost_histogram as bh

    values = [1]*9 + [91]
    wm = bh.accumulators.WeightedMean().fill(values, weight=2)
    print(f"{wm = }")

.. code:: text

    wm = WeightedMean(sum_of_weights=20, sum_of_weights_squared=40, value=10, variance=810)

You can add a `weight=` keyword when filling, with either a single value or a
matching length array of values.

You can call a WeightedMean with a value or with another WeightedMean to fill
inplace, as well.

Views
-----

Most of the accumulators (except Sum) support a View. This is what is returned from
a histogram when ``.view()`` is requested. This is a structured Numpy ndarray, with a few small
additions to make them easier to work with. Like a Numpy recarray, you can access the fields with
attributes; you can even access (but not set) computed attributes like ``.variance``. A view will
also return an accumulator instance if you select a single item.
