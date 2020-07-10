.. _usage-comparison:

Comparison with Boost.Histogram
===============================

``boost-histogram`` was based on the C++ library Boost.Histogram. In most ways,
it mimics the spirit of this library; if you learn to use one, you probably can use
the other. There are a few differences, however, mostly around adhering to Python
conventions:

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

Histogram make functions
   These functions, such as ``make_histogram`` and ``make_weighted_histogram``, are provided
   in Boost.Histogram to make the template syntax easier in C++14. In C++17, they are replaced
   by directly using the ``histogram`` constructor; the Python bindings are not limited by old
   templating syntax, and choose to only provide the newer spelling.

Custom components
   Many components in Boost.Histogram are configurable or replaceable at compile time; since
   Python code is precompiled, a comprehensive but static subset was selected for the Python bindings.

Changes
^^^^^^^

Naming
   The bindings follow modern Python conventions, with CamelCase for classes,
   etc. The Boost.Histogram library follows Boost conventions.

Serialization
   The Python bindings use a pickle-based binary serialization, so cannot read
   files saved in C++ using Boost.Serialize.

Properties
   Many methods in C++ are properties in Python. ``.axis(i)`` is replaced with ``.axes[i]``.

Indexing
  The Python bindings use standard Python indexing for selection and setting.
  You can recover the functionality of ``.at(i)`` at endpoints with
  ``bh.tag.at(i)``.

Renames
  The ``.rank()`` method is replaced by the ``.ndim`` property to match the common NumPy spelling.

Additions
^^^^^^^^^

Unified Histogram Indexing
   The Python bindings support UHI, a proposal to unify and simplify histogram
   indexing in Python.

Custom transforms
   Custom transforms are possible using Numba or a C pointer. In
   Boost.Histogram, you can use templating to make arbitrary transforms, so a
   run time transform is not as necessary (but may be added).

NumPy compatibility
   The Python bindings do several things to simplify NumPy compatibility.
