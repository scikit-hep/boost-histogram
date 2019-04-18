import pytest

import numpy as np
import boost.histogram as bh
from numpy.testing import assert_array_equal, assert_allclose

from boost.histogram.axis import regular_uoflow, regular_noflow

bins=100
ranges=(-1,1)
bins = np.asarray(bins).astype(np.int64)
ranges = np.asarray(ranges).astype(np.float64)

edges = np.linspace(ranges[0], ranges[1], bins+1)

np.random.seed(42)
vals = np.random.normal(size=[10000000]).astype(np.float32)

answer, _ = np.histogram(vals, bins=bins, range=ranges)

def test_numpy_perf_1d(benchmark):
    result, _ = benchmark(np.histogram, vals, bins=bins, range=ranges)
    assert_array_equal(result, answer)

def make_and_run_hist(hist, axes, vals, fill):
    histo = hist(axes)
    if fill is None:
        histo.fill(vals)
    elif fill < 0:
        histo.fill(vals, atomic=-fill)
    else:
        histo.fill(vals, threads=fill)

    return histo.view()

histax = (
        (bh.hist.any_int, regular_uoflow, None),
        (bh.hist.any_int, regular_noflow, None),
        (bh.hist.regular_int, regular_uoflow, None),
        (bh.hist.regular_noflow_int, regular_noflow, None),
        (bh.hist.regular_atomic_int, regular_uoflow, None),
        (bh.hist.regular_atomic_int, regular_uoflow, -4),
        (bh.hist.regular_int, regular_uoflow, 4),
        )

@pytest.mark.parametrize("hist, axis, fill", histax)
def test_1d(benchmark, hist, axis, fill):
    result = benchmark(make_and_run_hist, hist, [axis(bins, *ranges)], vals, fill)
    assert_allclose(result[:-1], answer[:-1], atol=2)
