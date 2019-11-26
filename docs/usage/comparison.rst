.. _usage-comparison:

Comparison with Boost.Histogram
===============================

``boost-histogram`` was based on the C++ library Boost.Histogram. In most ways,
it mimics the interface of this library; if you learn to use one, you probably can use
the other. A special high-compatibility mode is available with ``boost_histogram.cpp``,
as well, which reduces the changes to a bare minimum. There are a few differences, however:

Removals
^^^^^^^^

There are a few parts of the Boost.Histogram interface that are not bound. They are:

The call operator
   This is provided in C++ to allow single item filling, and was designed to mimic the
   accumulator syntax used elsewhere in Boost. It also works nicely with some STL
   algorithms. It was not provided in Python because using call to modify an object
   is not common in Python, using call makes duck-typing more dangerous, and single-item
   fills are not encouraged in Python due to poor performance. The ``.fill`` method from
   Boost.Histogram 1.72 is bound, however - this provides fast fills without the drawbacks.
   If you want to fill with a single item, Python's ``.fill`` does support single item fills.
   ``cpp`` mode does have a call operator.

Histogram make functions
   These functions, such as ``make_histogram`` and ``make_weighted_histogram``, are provided
   in Boost.Histogram to make the template syntax easier in C++14. In C++17, they are replaced
   by directly using the ``histogram`` constructor; the Python bindings are not limited by old
   templating syntax, and choose to only provide the newer spelling.
   ``cpp`` mode does have the make functions.

Custom components
   Many components in Boost.Histogram are configurable or replaceable at compile time; since
   Python code is precompiled, a comprehensive but static subset was selected for the Python bindings.

Changes
^^^^^^^

Serialization
   The Python bindings use a pickle based binary serialization, so cannot read files saved in C++ using Boost.Serialize.

Properties
   Many methods in C++ are properties in Python. In ``cpp`` mode, most of these remain methods, except for ``.metadata``,
   which needs to be a property to allow it to be set.


Additions
^^^^^^^^^

Unified Histogram Indexing
   The Python bindings support UHI, a proposal to unify and simplify histogram indexing in Python. Currently,
   this is disabled in ``cpp`` mode.

Numpy compatibility
   The Python bindings do several things to simplify Numpy compatibility.


