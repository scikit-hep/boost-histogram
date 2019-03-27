import pytest

import numpy as np
import histogram as bh

def test_make_regular_1D():
    hist_linear = bh.make_histogram(bh.axis.regular(1000,0,1))
    hist_atomic = bh.make_histogram(bh.axis.regular(1000,0,1),
                                    storage=bh.storage.dense_atomic_int())

    vals = np.random.rand(10000)

    hist_linear(vals)
    hist_atomic(vals)

    assert np.all(np.asarray(hist_linear) == np.asarray(hist_atomic))



