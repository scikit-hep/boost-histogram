.. _usage-indexing:

Indexing
========

Boost-histogram implements the UHI indexing protocol. You can read more about it on the `UHI Indexing <https://uhi.readthedocs.io/en/latest/indexing.html>`_ page.


Boost-histogram specific details
--------------------------------

Boost-histogram implements ``bh.loc``, ``builtins.sum``, ``bh.rebin``, ``bh.underflow``, and ``bh.overflow`` from the UHI spec. A ``bh.tag.at`` locator is provided as well, which simulates the Boost.Histogram C++ ``.at()`` indexing using the UHI locator protocol.

Boost-histogram allows "picking" using lists, similar to NumPy. If you select
with multiple lists, boost-histogram instead selects per-axis, rather than
group-selecting and reducing to a single axis, like NumPy does. You can use
``bh.loc(...)`` inside these lists.

Example::

    h = bh.histogram(
        bh.axis.Regular(10, 0, 1),
        bh.axis.StrCategory(["a", "b", "c"]),
        bh.axis.IntCategory([5, 6, 7]),
    )

    minihist = h[:, [bh.loc("a"), bh.loc("c")], [0, 2]]

    # Produces a 3D histgoram with Regular(10, 0, 1) x StrCategory(["a", "c"]) x IntCategory([5, 7])


This feature is considered experimental in boost-histogram 1.1.0. Removed bins are not added to the overflow bin currently.
