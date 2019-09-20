from .core.algorithm import shrink_and_rebin, slice_and_rebin, rebin, shrink, slice


def sum(histogram, flow=False):
    """Sum a histogram, optionally with flow bins"""
    return histogram.sum(flow)


def reduce(histogram, *args):
    "Reduce a histogram with 1 or more reduce options"
    return histogram.reduce(*args)
