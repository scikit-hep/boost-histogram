import pytest

import numpy as np
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

    assert np.all(np.asarray(hist_linear) == np.asarray(hist_atomic))

def test_atomic_fill_1D():
    hist_linear = bh.make_histogram(bh.axis.regular_uoflow(1000,0,1))
    hist_atomic = bh.make_histogram(bh.axis.regular_uoflow(1000,0,1),
                                    storage=bh.storage.atomic_int())

    vals = np.random.rand(10000)

    hist_linear(vals)

    with ThreadPoolExecutor(4) as pool:
        for i in range(4):
            pool.submit(hist_atomic, vals[i*2500:(i+1)*2500])

    assert np.all(np.asarray(hist_linear) == np.asarray(hist_atomic))
