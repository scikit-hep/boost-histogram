.. _usage-histogram:

Histogram
=========

The Histogram object is the core of boost-histogram.

Filling
^^^^^^^

You call ``.fill`` to fill. You must have one 1D array (or scalar value) per dimension. For maximum performance,
numeric arrays should be continuously laid out in memory, and either 64-bit floats or ints. If any other layouts or
numeric datatypes are supplied, a temporary copy will be made internally before filling.

All storages support a ``weight=`` parameter, and some storages support a ``sample=`` parameter. If supplied, they must be a scalar (applies to all items equally) or an iterable of scalars/1D arrays that matches the number of dimensions of the histogram.

The summing accumulators (not ``Mean()`` and ``WeightedMean())``) support threaded filling. Pass ``threads=N`` to the fill parameter to fill with ``N`` threads (and using 0 will select the number of virtual cores on your system). This is helpful only if you have a large number of entries compared to your number of bins, as all non-atomic storages will make copies for each thread, and then will recombine after the fill is complete.

Views
^^^^^

While Histograms do conform to the Python buffer protocol, the best way to get access to the contents of a histogram as a Numpy array is with ``.view()``. This way you can optionally pass ``flow=True`` to get the flow bins, and if you have an accumulator storage, you will get a View, which is a slightly augmented ndarrray subclass (see :ref:`usage-accumulators`).


Operations
^^^^^^^^^^

* ``h.rank``: The number of dimensions
* ``h.size or len(h)``: The number of bins

* ``+``: Add two histograms (storages must match types currently)
* ``*=``: Multiply by a scaler (not all storages) (``hist * scalar`` and ``scalar * hist`` supported too)
* ``/=``: Divide by a scaler (not all storages) (``hist / scalar`` supported too)
* ``[...]``: Access a bin or a range of bins (get or set) (see :ref:`usage-indexing`)

* ``.sum(flow=False)``: The total count of all bins
* ``.project(ax1, ax2, ...)``: Project down to listed axis (numbers)
* ``.to_numpy(flow=False)``: Convert to a Numpy style tuple (with or without under/overflow bins)
* ``.view(flow=False)``: Get a view on the bin contents (with or without under/overflow bins)
* ``.reset()``: Set counters to 0
* ``.empty(flow=False)``: Check to see if the histogram is empty (can check flow bins too if asked)
* ``.copy(deep=False)``: Make a copy of a histogram

* ``.axes``: Get the axes as a tuple-like (all properties of axes are available too)

    * ``.axes[0]``: Get the 0th axis

    * ``.axes.edges``: The lower values as a broadcasting-ready array
    * ``.axes.centers``: The centers of the bins broadcasting-ready array
    * ``.axes.widths``: The bin widths as a broadcasting-ready array
    * ``.axes.metadata``: A tuple of the axes metadata

    * ``.axes.size``: A tuple of the axes sizes (size without flow)
    * ``.axes.extent``: A tuple of the axes extents (size with flow)

    * ``.axes.bin(*args)``: Returns the bin edges as a tuple of pairs (continuous axis) or values (describe)
    * ``.axes.index(*args)``: Returns the bin index at a value for each axis
    * ``.axes.value(*args)``: Returns the bin value at an index for each axis

Saving a Histogram
^^^^^^^^^^^^^^^^^^

You can save a histogram using pickle:

.. code:: python3

    import pickle

    with open('file.pkl', 'wb') as f:
        pickle.dump(h, f)

    with open('file.pkl', 'rb') as f:
        h2 = pickle.load(f)

    assert h == h2

Special care was taken to ensure that this is fast and efficient.  Please use
the latest version of the Pickle protocol you feel comfortable using; you
cannot use version 0, the version that is default on Python 2. The most recent
versions provide performance benefits.

You can nest this in other Python structures, like dictionaries, and save those instead.
