.. _usage-indexing:

Indexing
========


This is the design document for Unified Histogram Indexing (UHI).  Much of the
original plan is now implemented in boost-histogram.  Other histogramming
libraries can implement support for this as well, and the "tag" functors, like
``sum`` and ``loc`` can be used between libraries.

Syntax
------

The following examples assume you have imported ``loc``, ``sum``, ``rebin``,
``underflow``, and ``overflow`` from boost-histogram or any other library that
implements UHI.

Access:
^^^^^^^

.. code:: python3

   v = h[b]          # Returns bin contents, indexed by bin number
   v = h[loc(b)]     # Returns the bin containing the value
   v = h[loc(b) + 1] # Returns the bin above the one containing the value
   v = h[underflow]  # Underflow and overflow can be accessed with special tags

Slicing:
^^^^^^^^

.. code:: python3

   h == h[:]             # Slice over everything
   h2 = h[a:b]           # Slice of histogram (includes flow bins)
   h2 = h[:b]            # Leaving out endpoints is okay
   h2 = h[loc(v):]       # Slices can be in data coordinates, too
   h2 = h[::rebin(2)]    # Modification operations (rebin)
   h2 = h[a:b:rebin(2)]  # Modifications can combine with slices
   h2 = h[::sum]         # Projection operations # (name may change)
   h2 = h[a:b:sum]       # Adding endpoints to projection operations
   h2 = h[0:len:sum]     #   removes under or overflow from the calculation
   h2 = h[v, a:b]        #   A single value v is like v:v+1:sum
   h2 = h[a:b, ...]      # Ellipsis work just like normal numpy

Setting
^^^^^^^

.. code:: python3

   # Single values
   h[b] = v         # Returns bin contents, indexed by bin number
   h[loc(b)] = v    # Returns the bin containing the value
   h[underflow] = v # Underflow and overflow can be accessed with special tags

   h[...] = array(...) # Setting with an array or histogram sets the contents if the sizes match
                       # Overflow can optionally be included if endpoints are left out
                       # The number of dimensions for non-scalars should match (broadcasting works normally otherwise)

All of this generalizes to multiple dimensions. ``loc(v)`` could return
categorical bins, but slicing on categories would (currently) not be
allowed. These all return histograms, so flow bins are always preserved
- the one exception is projection; since this removes an axis, the only
use for the slice edges is to be explicit on what part you are
interested for the projection. So an explicit (non-empty) slice here
will case the relevant flow bin to be excluded.

``loc``, ``project``, and ``rebin`` all live inside the histogramming
package (like boost-histogram), but are completely general and can be created by a
user using an explicit API (below). ``underflow`` and ``overflow`` also
follow a general API.

One drawback of the syntax listed above is that it is hard to select an action
to run on an axis or a few axes out of many. For this use case, you can pass a
dictionary to the index, and that has the syntax ``{axis:action}``. The actions
are slices, and follow the rules listed above. This looks like:

.. code:: python3

    h[{0: slice(None, None, bh.rebin(2))}] # rebin axis 0 by two
    h[{1: slice(0, bh.loc(3.5))}]          # slice axis 1 from 0 to the data coordinate 3.5
    h[{7: slice(0, 2, bh.rebin(4))}]       # slice and rebin axis 7


If you don't like manually building slices, you can use the `Slicer()` utility
to recover the original slicing syntax inside the dict:

.. code:: python3

    s = bh.tag.Slicer()

    h[{0: s[::bh.rebin(2)]}]   # rebin axis 0 by two
    h[{1: s[0:bh.loc(3.5)]}]   # slice axis 1 from 0 to the data coordinate 3.5
    h[{7: s[0:2:bh.rebin(4)]}] # slice and rebin axis 7



Invalid syntax:
^^^^^^^^^^^^^^^

.. code:: python3

   h[1.0] # Floats are not allowed, just like numpy
   h[::2] # Skipping is not (currently) supported
   h[..., None] # None == np.newaxis is not supported

Rejected proposals or proposals for future consideration, maybe ``hist``-only:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python3

   h2 = h[1.0j:2.5j + 1] # Adding a j suffix to a number could be used in place of `loc(x)`
   h2 = h[1.0] # Floats in place of `loc(x)`: too easy to make a mistake

Examples
--------


For a histogram, the slice should be thought of like this:

.. code:: python3

   histogram[start:stop:action]

The start and stop can be either a bin number (following Python rules),
or a callable; the callable will get the axis being acted on and should
return an extended bin number (``-1`` and ``len(ax)`` are flow bins). A
provided callable is ``bh.loc``, which converts from axis data
coordinates into bin number.

The final argument, ``action``, is special. A general API is being
worked on, but for now, ``bh.sum`` will “project out” or “integrate
over” an axes, and ``bh.rebin(n)`` will rebin by an integral factor.
Both work correctly with limits; ``bh.sum`` will remove flow bins if
given a range. ``h[0:len:bh.sum]`` will sum without the flow bins.

Here are a few examples that highlight the functionality of UHI:

Example 1:
^^^^^^^^^^

You want to slice axis 0 from 0 to 20, axis 1 from .5 to 1.5 in data
coordinates, axis 2 needs to have double size bins (rebin by 2), and
axis 3 should be summed over. You have a 4D histogram.

Solution:

.. code:: python3

   ans = h[:20, bh.loc(-.5):bh.loc(1.5), ::bh.rebin(2), ::bh.sum]

Example 2:
^^^^^^^^^^

You want to set all bins above 4.0 in data coordinates to 0 on a 1D
histogram.

Solution:

.. code:: python3

   h[bh.loc(4.0):] = 0

You can set with an array, as well. The array can either be the same
length as the range you give, or the same length as the range +
under/overflows if the range is open ended (no limit given). For
example:

.. code:: python3

   h = bh.Histogram(bh.axis.Regular(10, 0, 1))
   h[:] = np.ones(10) # underflow/overflow still 0
   h[:] = np.ones(12) # underflow/overflow now set too

Note that for clarity, while basic Numpy broadcasting is supported,
axis-adding broadcasting is not supported; you must set a 2D histogram
with a 2D array or a scalar, not a 1D array.

Example 3:
^^^^^^^^^^

You want to sum from -infinity to 2.4 in data coordinates in axis 1,
leaving all other axes alone. You have an ND histogram, with N >= 2.

Solution:

.. code:: python3

   ans = h[:, :bh.loc(2.4):bh.sum, ...]

Notice that last example could be hard to write if the axis number, 1 in
this case, was large or programmatically defined. In these cases, you
can pass a dictionary of ``{axis:slice}`` into the indexing operation. A
shortcut to quickly generate slices is provided, as well:

.. code:: python3

   ans = h[{1: slice(None,bh.loc(2.4),bh.sum)}]

   # Identical:
   s = bh.tag.Slicer()
   ans = h[{1: s[:bh.loc(2.4):bh.sum]}]

Example 4:
^^^^^^^^^^

You want the underflow bin of a 1D histogram.

Solution:

.. code:: python3

   val = h1[bh.underflow]





--------------

Details
-------


Axis indexing
^^^^^^^^^^^^^

TODO: Possibly extend to axes. Would follow the 1D cases above.

Implementation notes
^^^^^^^^^^^^^^^^^^^^

loc, rebin, and sum are *not* unique tags, or special types, but rather
APIs for classes. New versions of these could be added, and
implementations could be shared among Histogram libraries. For clarity,
the following code is written in Python 3.6+. `Prototype
here <https://gist.github.com/henryiii/d545a673ea2b3225cb985c9c02ac958b>`__.
`Extra doc
here <https://docs.google.com/document/d/1bJKA7Y0QXf46w53UFizJ4bnZlVIkb4aCqx6m2hoN0HM/edit#heading=h.jvegm6z8f387>`__.

Note that the API comes in two forms; the ``__call__``/``__new__`` operator
form is more powerful, slower, optional, and is currently not supported by
boost-histogram.  A fully conforming UHI implementation must allow the tag form
without the operators.

Basic implementation (WIP):

.. code:: python3

   class loc:
       "When used in the start or stop of a Histogram's slice, x is taken to be the position in data coordinates."
       def __init__(self, value, offset):
           self.value = value
           self.offset = offest

       # supporting __add__ and __sub__ also recommended

       def __call__(self, axis):
           return axis.index(self.value) + self.offset

   # Other flags, such as callable functions, could be added and detected later.

   # UHI will perform a maximum performance sum when python's sum is encountered

   def underflow(axis):
       return -1
   def overflow(axis):
       return len(axis)


   class rebin:
       """
       When used in the step of a Histogram's slice, rebin(n) combines bins,
       scaling their widths by a factor of n. If the number of bins is not
       divisible by n, the remainder is added to the overflow bin.
       """
       def __init__(self, factor):
           # Items with .factor are specially treated in boost-histogram,
           # performing a high performance rebinning
           self.factor = factor

       # Optional and not used by boost-histogram
       def __call__(self, binning, axis, counts):
           factor = self.factor
           if isinstance(binning, Regular):
               indexes = (numpy.arange(0, binning.num, factor),)

               num, remainder = divmod(binning.num, factor)
               high, hasover = binning.high, binning.hasover

               if binning.hasunder:
                   indexes[0][:] += 1
                   indexes = ([0],) + indexes

               if remainder == 0:
                   if binning.hasover:
                       indexes = indexes + ([binning.num + int(binning.hasunder)],)
               else:
                   high = binning.left(indexes[-1][-1])
                   hasover = True

               binning = Regular(num, binning.low, high, hasunder=binning.hasunder, hasover=hasover)
               counts = numpy.add.reduceat(counts, numpy.concatenate(indexes), axis=axis)
               return binning, counts

           else:
               raise NotImplementedError(type(binning))
