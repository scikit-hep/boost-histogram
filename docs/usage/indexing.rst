.. _usage-indexing:

Indexing
========

Boost-histogram implements the UHI indexing protocol. You can read more about it on the `UHI Indexing <https://uhi.readthedocs.io/en/latest/indexing.html>`_ page.


Boost-histogram specific details
--------------------------------

Boost-histogram implements ``bh.loc``, ``builtins.sum``, ``bh.rebin``, ``bh.underflow``, and ``bh.overflow`` from the UHI spec. A ``bh.tag.at`` locator is provided as well, which simulates the Boost.Histogram C++ ``.at()`` indexing using the UHI locator protocol.
