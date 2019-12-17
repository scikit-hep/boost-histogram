.. _usage-histogram:

Histogram
=========

The Histogram object is the core of boost-histogram.

Filling
^^^^^^^

You call ``.fill`` to fill. You must have one 1D array (or scalar value) per dimension. For maximum performance,
numeric arrays should be continuously laid out in memory, and either 64-bit floats or ints. If any other layouts or
numeric datatypes are supplied, a temporary copy will be made internally before filling.

All storages support a `weight=` parameter, and some storages support a `sample=` parameter. If supplied, they must be a scalar (applies to all items equally) or an iterable of scalars/1D arrays that matches the number of dimensions of the histogram.

Views
^^^^^

While Histograms do conform to the Python buffer protocol, the best way to get access to the contents of a histogram as a Numpy array is with ``.view()``. This way you can optionally pass ``flow=True`` to get the flow bins, and if you have an accumulator storage, you will get a View, which is a slightly augmented ndarrray subclass (see :ref:`usage-accumulators`).


Operations
^^^^^^^^^^

* ``h.rank``: The number of dimensions
* ``h.size or len(h)``: The number of bins
* ``.reset()``: Set counters to 0
* ``+``: Add two histograms
* ``*=``: Multiply by a scaler (not all storages) (``hist * scalar`` and ``scalar * hist`` supported too)
* ``/=``: Divide by a scaler (not all storages) (``hist / scalar`` supported too)
* ``.to_numpy(flow=False)``: Convert to a Numpy style tuple (with or without under/overflow bins)
* ``.view(flow=False)``: Get a view on the bin contents (with or without under/overflow bins)
* ``.axes``: Get the axes
    * ``.axes[0]``: Get the 0th axis
    * ``.axes.edges``: The lower values as a broadcasting-ready array
    * All other properties of axes available here, too
* ``.sum(flow=False)``: The total count of all bins
* ``.project(ax1, ax2, ...)``: Project down to listed axis (numbers)
