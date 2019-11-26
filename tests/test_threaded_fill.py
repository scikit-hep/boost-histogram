import pytest

import boost_histogram as bh
from threaded import thread_fill, classic_fill, atomic_fill

import numpy as np
from numpy.testing import assert_array_equal
from functools import partial


@pytest.mark.benchmark(group="threaded-fill-1d")
@pytest.mark.parametrize("method", [thread_fill, classic_fill, atomic_fill])
def test_threads(benchmark, method):
    axes = [bh.axis.Regular(1000, 0, 1)]
    hist_linear = bh.Histogram(*axes)

    vals = np.random.rand(100000)
    hist_linear.fill(vals)

    method_th = partial(method, axes, 4)
    hist_atomic = benchmark(method_th, vals)

    assert_array_equal(hist_linear, hist_atomic)


@pytest.mark.parametrize("threads", [1, 2, 4, 7])
def test_threaded_builtin(threads):
    axes = [bh.axis.Regular(1000, 0, 1)]
    hist_atomic1 = bh.Histogram(*axes, storage=bh.storage.AtomicInt64())

    vals = np.random.rand(10000)

    hist_atomic1.fill(vals)
    hist_atomic2 = atomic_fill(axes, 4, vals)

    assert_array_equal(hist_atomic1, hist_atomic2)
