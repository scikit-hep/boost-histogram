.. _usage-analyses:

Analyses examples
=================

Bool and category axes
----------------------

Taken together, the flexibility in axes and the tools to easily sum over
axes can be applied to transform the way you approach analysis with
histograms. For example, let’s say you are presented with the following
data in a 3xN table:

============== ========================
Data           Details
============== ========================
``value``
``is_valid``   True or False
``run_number`` A collection of integers
============== ========================

In a traditional analysis, you might bin over ``value`` where
``is_valid`` is True, and then make a collection of histograms, one for
each run number. With boost-histogram, you can make a single histogram,
and use an axis for each:

.. code:: python3

   value_ax = bh.axis.Regular(100, -5, 5)
   bool_ax = bh.axis.Integer(0, 2, underflow=False, overflow=False)
   run_number_ax = bh.axis.IntCategory([], growth=True)

Now, you can use these axes to create a single histogram that you can
fill. If you want to get a histogram of all run numbers and just the
True ``is_valid`` selection, you can use a ``sum``:

.. code:: python3

   h1 = hist[:, True, ::bh.sum]

You can expand this example to any number of dimensions, boolean flags,
and categories.
