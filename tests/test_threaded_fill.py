import pytest

import numpy as np
from numpy.testing import assert_array_equal
import boost.histogram as bh

from concurrent.futures import ThreadPoolExecutor
from functools import reduce, partial
from operator import add

def mk_and_fill(vals):
    hist = bh._make_histogram(bh.axis.regular(1000,0,1))
    hist.fill(vals)
    return hist

def thread_fill(threads, vals):
    with ThreadPoolExecutor(threads) as pool:
        results = pool.map(mk_and_fill, [vals[i*25000:(i+1)*25000] for i in range(4)])

    return reduce(add, results)

def atomic_fill(threads, vals):
    hist = bh._make_histogram(bh.axis.regular(1000,0,1), storage=bh.storage.atomic_int())
    with ThreadPoolExecutor(threads) as pool:
        pool.map(hist.fill, [vals[i*25000:(i+1)*25000] for i in range(4)])
    return hist

def classic_fill(threads, vals):
    hist = bh._make_histogram(bh.axis.regular(1000,0,1))
    hist.fill(vals)
    return hist


@pytest.mark.benchmark(group="threaded-fill")
@pytest.mark.parametrize("method", [thread_fill, classic_fill, atomic_fill])
def test_threads(benchmark, method):
    hist_linear = bh._make_histogram(bh.axis.regular_uoflow(1000,0,1))

    vals = np.random.rand(100000)
    hist_linear.fill(vals)

    method_th = partial(method, 4)
    hist_atomic = benchmark(method_th, vals)

    assert_array_equal(hist_linear, hist_atomic)


@pytest.mark.parametrize("threads", [1,2,4,7])
def test_threaded_builtin(threads):
    hist_atomic1 = bh._make_histogram(bh.axis.regular_uoflow(1000,0,1), storage=bh.storage.atomic_int())

    vals = np.random.rand(10000)

    hist_atomic1.fill(vals)
    hist_atomic2 = atomic_fill(4, vals)

    assert_array_equal(hist_atomic1, hist_atomic2)
