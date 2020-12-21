.. _usage-plotting:

Plotting
========

Boost-histogram does not contain plotting functions - this is outside of the
scope of the project, which is histogram filling and manipulation. However, it
does follow ``PlottableProtocol``, as listed below. Any plotting library that
accepts an object that follows the ``PlottableProtocol`` can plot boost-histogram
objects.

Using the protocol:

Plotters should only depend on the methods and attributes listed below. In short, they are:

* ``h.kind``: The bh.Kind of the histogram (COUNT or MEAN)
* ``h.values()``: The value (as given by the kind)
* ``h.variances()``: The variance in the value (None if an unweighed histogram was filled with weights)
* ``h.counts()``: The effective counts
* ``h.axes``: A Sequence of axes

Axes have:

* ``ax[i]``: A sequence of lower, upper bin, or the discrete bin value (integer or sting)
* ``len(ax)``: The number of bins
* ``ax.traits.circular``: True if circular
* ``ax.traits.discrete``: True if discrete (Integer or Category axes)

Plotters should see if ``.counts()`` is None; no boost-histogram objects currently
return None, but a future storage or different library could.

Also check ``.variances``; if not None, this storage holds variance information and
error bars should be included. Boost-histogram histograms will return something
unless they know that this is an invalid assumption (a weighted fill was made
on an unweighted histogram).

The full protocol version 1 follows:

.. literalinclude:: ../../tests/plottable.py
   :language: python
