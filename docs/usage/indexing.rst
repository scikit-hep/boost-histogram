Indexing
========

This is the design document for Unified Histogram Indexing (UHI).  Much of the original plan is now implemented in boost-histogram.
Other histogramming libraries can implement support for this as well, and the "tag" functors, like ``project`` and ``loc`` can be
used between libraries.

The following examples assume you have imported ``loc``, ``project``, ``rebin``, and ``end`` from boost-histogram or any other
library that implements UHI.

Access:
^^^^^^^

.. code:: python

   v = h[b] # Returns bin contents, indexed by bin number
   v = h[loc(b)] # Returns the bin containing the value

Slicing:
^^^^^^^^

.. code:: python

   h == h[:]             # Slice over everything
   h2 = h[a:b]           # Slice of histogram (includes flow bins)
   h2 = h[:b]            # Leaving out endpoints is okay
   h2 = h[loc(v):]       # Slices can be in data coordinates, too
   h2 = h[::project]     # Projection operations
   h2 = h[::rebin(2)]    # Modification operations (rebin)
   h2 = h[a:b:rebin(2)]  # Modifications can combine with slices
   h2 = h[a:b, ...]      # Ellipsis work just like normal numpy

   # Not yet supported!
   h2 = h[a:b:project] # Adding endpoints to projection operations removes under or overflow from the calculation
   h2 = h[0:end:project] # Special end functor (TBD)
   h2 = h[0:len(h2.axis(0)):project] # Projection without flow bins

Setting (Not yet supported)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   h[...] = np.ndarray(...) # Setting with an array or histogram sets the contents if the sizes match
                            # Overflow can optionally be included
   h[b] = v                 # Single values work too

All of this generalizes to multiple dimensions. ``loc(v)`` could return
categorical bins, but slicing on categories would (currently) not be
allowed. These all return histograms, so flow bins are always preserved
- the one exception is projection; since this removes an axis, the only
use for the slice edges is to be explicit on what part you are
interested for the projection. So an explicit (non-empty) slice here
will case the relevant flow bin to be excluded (not currently supported).

``loc``, ``project``, and ``rebin`` all live inside the histogramming
package (like boost-histogram), but are completely general and can be created by a
user using an explicit API (below).

Invalid syntax:
^^^^^^^^^^^^^^^

.. code:: python

   h[v, a:b] # You cannot mix slices and bin contents access (h is an integer)
   h[1.0] # Floats are not allowed, just like numpy
   h[::2] # Skipping is not (currently) supported
   h[..., None] # None == np.newaxis is not supported

Rejected proposals or proposals for future consideration, maybe ``hist``-only:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   h2 = h[1j:2j] # Adding a j suffix to a number could be used in place of `loc(x)`
   h2 = h[1.0] # Floats in place of `loc(x)`: too easy to make a mistake

--------------

Implementation notes
--------------------

loc, rebin, and project are *not* unique tags, or special types, but rather
APIs for classes. New versions of these could be added, and
implementations could be shared among Histogram libraries. For clarity,
the following code is written in Python 3.6+. `Prototype
here <https://gist.github.com/henryiii/d545a673ea2b3225cb985c9c02ac958b>`__.
`Extra doc
here <https://docs.google.com/document/d/1bJKA7Y0QXf46w53UFizJ4bnZlVIkb4aCqx6m2hoN0HM/edit#heading=h.jvegm6z8f387>`__.

Note that the API comes in two forms; the ``__call__`` operator form is more powerful, slower, optional, and is not supported by boost.histogram.

Basic implementation (WIP):

.. code:: python

   class loc:
       "When used in the start or stop of a Histogram's slice, x is taken to be the position in data coordinates."
       def __init__(self, x):
           self.value = x

   # Other flags, such as callable functions, could be added and detected later.

   class project:
       "When used in the step of a Histogram's slice, project sums over and eliminates what remains of the axis after slicing."
       projection = True
       def __new__(cls, binning, axis, counts):
         return None, numpy.add.reduce(counts, axis=axis)



   class rebin:
       "When used in the step of a Histogram's slice, rebin(n) combines bins, scaling their widths by a factor of n. If the number of bins is not divisible by n, the remainder is added to the overflow bin."
       projection = False
       def __init__(self, factor):
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
