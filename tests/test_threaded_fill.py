import pytest

import numpy as np
from numpy.testing import assert_array_equal
import boost.histogram as bh

from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from operator import add

def mk_and_fill(vals):
    hist = bh._make_histogram(bh.axis.regular(1000,0,1))
    hist.fill(vals)
    return hist

def thread_fill(vals):
    with ThreadPoolExecutor(4) as pool:
        results = pool.map(mk_and_fill, [vals[i*25000:(i+1)*25000] for i in range(4)])

    return reduce(add, results)


def classic_fill(vals):
    hist = bh._make_histogram(bh.axis.regular(1000,0,1))
    hist.fill(vals)
    return hist


def hardcoded_fill(vals):
    hist = bh._make_histogram(bh.axis.regular(1000,0,1))
    hist.fill(vals, threads=4)
    return hist


@pytest.mark.benchmark(group="threaded-fill")
@pytest.mark.parametrize("method", [thread_fill, classic_fill, hardcoded_fill])
def test_threads(benchmark, method):
    hist_linear = bh._make_histogram(bh.axis.regular_uoflow(1000,0,1))

    vals = np.random.rand(100000)
    hist_linear.fill(vals)

    hist_atomic = benchmark(method, vals)

    assert_array_equal(hist_linear, hist_atomic)


@pytest.mark.parametrize("threads", [1,2,4,7])
def test_threaded_builtin(threads):
    hist_atomic1 = bh._make_histogram(bh.axis.regular_uoflow(1000,0,1))
    hist_atomic2 = bh._make_histogram(bh.axis.regular_uoflow(1000,0,1))

    vals = np.random.rand(10000)

    hist_atomic1.fill(vals)
    hist_atomic2.fill(vals, threads=threads)

    assert_array_equal(hist_atomic1, hist_atomic2)
