import pytest

import boost.histogram as bh
import numpy as np
from numpy.testing import assert_allclose
import itertools

def test_1d_center():
    h = bh.histogram(bh.axis.regular(4,0,1))

    results = np.r_[.125:.875:4j]

    for ind, answer in zip(h.indexed(), results):
        a_bin, = ind.bins()
        assert a_bin.center() == answer
        center, = ind.centers()
        assert center == answer


# Any axis has a special replacement for internal_view
def test_2d_any_center():

    # This iterates in the opposite order as boost-histogram
    results = itertools.product(np.r_[0:4], np.r_[.125:.875:4j])

    h = bh.histogram(bh.axis.regular(4,0,1), bh.axis.integer(0,4))

    for ind, answer in zip(h.indexed(), results):
        a, b = ind.centers()
        assert b == answer[0]
        assert a == answer[1]

        a, b = ind.bins()
        assert b.center() == answer[0]
        assert a.center() == answer[1]


def test_2d_function():

    gridx, gridy = np.mgrid[.125:.875:4j, .25:1.75:4j]

    def f(x, y):
        return 1+x**3 - 7*y**4

    result = f(gridx, gridy)


    h = bh.histogram(bh.axis.regular(4,0,1), bh.axis.regular(4,0,2), storage=bh.storage.double())

    for ind in h.indexed():
        ind.content = f(*ind.centers())

    assert_allclose(h, result)

def benchmark_1d_indexed(func):
    hist = bh.histogram(bh.axis.regular(100,0,1), storage=bh.storage.double())
    for ind in hist.indexed():
        ind.content = func(*ind.centers())
    return hist

def benchmark_1d_ufunc(func):
    hist = bh.histogram(bh.axis.regular(100,0,1), storage=bh.storage.double())
    view = np.asarray(hist)
    vals = func(hist.centers())
    view[:] = func(hist.centers())
    return hist
    # TODO: .view() seems to make a copy

def func(x):
    return x**2

@pytest.mark.benchmark(group='1D indexed')
def test_1d_indexed(benchmark):
    res = benchmark(benchmark_1d_indexed, func)

    hist = benchmark_1d_ufunc(func)
    assert_allclose(hist, res)

@pytest.mark.benchmark(group='1D indexed')
def test_1d_ufunc(benchmark):
    benchmark(benchmark_1d_ufunc, func)
