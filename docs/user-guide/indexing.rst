.. _usage-indexing:

Indexing
========

Boost-histogram implements the UHI indexing protocol. You can read more about it on the `UHI Indexing <https://uhi.readthedocs.io/en/latest/indexing.html>`_ page.


Boost-histogram specific details
--------------------------------

Boost-histogram implements ``bh.loc``, ``builtins.sum``, ``bh.rebin``, ``bh.underflow``, and ``bh.overflow`` from the UHI spec. A ``bh.tag.at`` locator is provided as well, which simulates the Boost.Histogram C++ ``.at()`` indexing using the UHI locator protocol.

Boost-histogram allows "picking" using lists, similar to NumPy. If you select with multiple lists, boost-histogram instead selects per-axis, rather than group-selecting and reducing to a single axis, like NumPy does. You can use ``bh.loc(...)`` inside these lists.

Example::

    h = bh.Histogram(
        bh.axis.Regular(10, 0, 1),
        bh.axis.StrCategory(["a", "b", "c"]),
        bh.axis.IntCategory([5, 6, 7]),
    )

    minihist = h[:, [bh.loc("a"), bh.loc("c")], [0, 2]]

    # Produces a 3D histogram with Regular(10, 0, 1) x StrCategory(["a", "c"]) x IntCategory([5, 7])


This feature is considered experimental in boost-histogram 1.1.0. Removed bins are not added to the overflow bin currently.


Vectorized indexing
-------------------

While *lists* select per-axis (as described above), *NumPy integer arrays* are
treated as vectorized fancy indexing, following NumPy's broadcasting rules. This
gathers (or scatters, when assigning) many individual cells in a single
operation, instead of building a new histogram. It is much faster than looping
over scalar indices, which matters for high-dimensional histograms with many
categories.

Example::

    import numpy as np

    h = bh.Histogram(
        bh.axis.IntCategory(range(100)),
        bh.axis.IntCategory(range(100)),
        bh.axis.Regular(50, 0, 1),
    )

    datasets = np.array([3, 7, 42])
    categories = np.array([1, 1, 9])

    # Gather the Regular-axis contents for three (dataset, category) pairs at once
    values = h[datasets, categories, :]  # shape (3, 50)

    # Assignment works the same way
    h[datasets, categories, :] = 0

The array indices use the same (non-flow) numbering as scalar indexing, and you
can mix them with integers, ``bh.loc(...)``, and plain integer slices. This path
covers the common case directly; for anything more advanced (rebinning,
``sum``-projection, or locator-based slices alongside arrays) index ``.view()``
explicitly instead.
