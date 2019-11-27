from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

__all__ = (
    "shrink_and_rebin",
    "shrink",
    "slice_and_rebin",
    "rebin",
    "shrink",
    "slice",
    "sum",
    "reduce",
    "empty",
    "reduce",
    "project",
)

from .._core.algorithm import shrink_and_rebin, slice_and_rebin, rebin, shrink, slice

shrink_and_rebin.__module__ = "boost_histogram.cpp"
slice_and_rebin.__module__ = "boost_histogram.cpp"
rebin.__module__ = "boost_histogram.cpp"
shrink.__module__ = "boost_histogram.cpp"
slice.__module__ = "boost_histogram.cpp"


def sum(histogram, flow=False):
    """Sum a histogram, optionally with flow bins"""
    return histogram._sum(flow)


def reduce(histogram, *args):
    "Reduce a histogram with 1 or more reduce options"
    return histogram._reduce(*args)


def empty(histogram, flow=False):
    """Check to see if a histogram is empty, optionally with flow bins"""
    return histogram._empty(flow)


def reduce(histogram, *args):
    """
    Reduce based on one or more reduce_option's.
    """

    return histogram._reduce(*args)


def project(histogram, *args):
    """
    Project to a single axis or several axes on a multidiminsional histogram.
    Provided a list of axis numbers, this will produce the histogram over those
    axes only. Flow bins are used if available.
    """

    return histogram._project(*args)
