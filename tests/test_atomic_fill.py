import pytest

import numpy as np
from numpy.testing import assert_array_equal
import boost.histogram as bh

from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from operator import add

def test_make_regular_1D():
    hist_linear = bh.make_histogram(bh.axis.regular_uoflow(1000,0,1))
    hist_atomic = bh.make_histogram(bh.axis.regular_uoflow(1000,0,1),
                                    storage=bh.storage.atomic_int())

    vals = np.random.rand(10000)

    hist_linear(vals)
    hist_atomic(vals)

    assert_array_equal(hist_linear, hist_atomic)

def test_atomic_fill_1D():
    hist_linear = bh.make_histogram(bh.axis.regular_uoflow(1000,0,1))
    hist_atomic = bh.make_histogram(bh.axis.regular_uoflow(1000,0,1),
                                    storage=bh.storage.atomic_int())

    vals = np.random.rand(10000)

    hist_linear(vals)

    with ThreadPoolExecutor(4) as pool:
        for i in range(4):
            pool.submit(hist_atomic, vals[i*2500:(i+1)*2500])

    assert_array_equal(hist_linear, hist_atomic)

@pytest.mark.parametrize("threads", [1,2,4,7])
def test_atomic_builtin(threads):
    hist_atomic1 = bh.make_histogram(bh.axis.regular_uoflow(1000,0,1),
                                     storage=bh.storage.atomic_int())
    hist_atomic2 = bh.make_histogram(bh.axis.regular_uoflow(1000,0,1),
                                     storage=bh.storage.atomic_int())

    vals = np.random.rand(10000)

    hist_atomic1(vals)
    hist_atomic2.atomic_fill(threads, vals)

    assert_array_equal(hist_atomic1, hist_atomic2)


@pytest.mark.parametrize("threads", [1,2,4,7])
def test_threaded_builtin(threads):
    hist_atomic1 = bh.make_histogram(bh.axis.regular_uoflow(1000,0,1))
    hist_atomic2 = bh.make_histogram(bh.axis.regular_uoflow(1000,0,1))

    vals = np.random.rand(10000)

    hist_atomic1(vals)
    hist_atomic2.threaded_fill(threads, vals)

    assert_array_equal(hist_atomic1, hist_atomic2)
