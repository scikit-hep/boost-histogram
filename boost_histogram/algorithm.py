from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from ._core.algorithm import shrink_and_rebin, slice_and_rebin, rebin, shrink, slice

for cls in (shrink_and_rebin, slice_and_rebin, rebin, shrink, slice):
    cls.__module__ = "boost_histogram.algorithm"

del cls
