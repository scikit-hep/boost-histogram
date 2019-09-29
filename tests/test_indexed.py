import pytest
from pytest import approx

import boost_histogram as bh
import numpy as np
from numpy.testing import assert_allclose
import itertools


def test_1d_center():
    h = bh.histogram(bh.axis.regular(2, 0, 1))

    bins = [(0, 0.5), (0.5, 1)]
    centers = [0.25, 0.75]

    for ind, bin_ref, center_ref in zip(h.indexed(), bins, centers):
        bin, = ind.bins()
        assert bin == approx(bin_ref)
        center, = ind.centers()
        assert center == approx(center_ref)


# Any axis has a special replacement for internal_view
def test_2d_center():

    h = bh.histogram(bh.axis.regular(2, 0, 1), bh.axis.integer(0, 2))

    bins = [((0, 0.5), 0), ((0.5, 1), 0), ((0, 0.5), 1), ((0.5, 1), 1)]
    centers = [(0.25, 0.5), (0.75, 0.5), (0.25, 1.5), (0.75, 1.5)]

    for ind, bins_ref, centers_ref in zip(h.indexed(), bins, centers):
        assert ind.bins()[0] == approx(bins_ref[0])
        assert ind.bins()[1] == approx(bins_ref[1])
        assert ind.centers() == approx(centers_ref)


def test_2d_function():

    gridx, gridy = np.mgrid[0.125:0.875:4j, 0.25:1.75:4j]

    def f(x, y):
        return 1 + x ** 3 - 7 * y ** 4

    result = f(gridx, gridy)

    h = bh.histogram(
        bh.axis.regular(4, 0, 1), bh.axis.regular(4, 0, 2), storage=bh.storage.double()
    )

    for ind in h.indexed():
        ind.content = f(*ind.centers())

    assert_allclose(h, result)


def benchmark_1d_indexed(func):
    hist = bh.histogram(bh.axis.regular(100, 0, 1), storage=bh.storage.double())
    for ind in hist.indexed():
        ind.content = func(*ind.centers())
    return hist


def benchmark_1d_ufunc(func):
    hist = bh.histogram(bh.axis.regular(100, 0, 1), storage=bh.storage.double())
    view = np.asarray(hist)
    vals = func(hist.centers())
    view[:] = func(hist.centers())
    return hist
    # TODO: .view() seems to make a copy


def func(x):
    return x ** 2


@pytest.mark.benchmark(group="1D indexed")
def test_1d_indexed(benchmark):
    res = benchmark(benchmark_1d_indexed, func)

    hist = benchmark_1d_ufunc(func)
    assert_allclose(hist, res)


@pytest.mark.benchmark(group="1D indexed")
def test_1d_ufunc(benchmark):
    benchmark(benchmark_1d_ufunc, func)
