import pytest

import numpy as np
from numpy.testing import assert_array_equal

import boost.histogram as bh
from boost.histogram.axis import regular_uoflow, regular_noflow

bins=(100, 100)
ranges=((-1,1),(-1,1))
bins = np.asarray(bins).astype(np.int64)
ranges = np.asarray(ranges).astype(np.float64)

edges = (np.linspace(ranges[0,0], ranges[0,1], bins[0]+1),
         np.linspace(ranges[1,0], ranges[1,1], bins[1]+1))

np.random.seed(42)
vals = np.random.normal(size=[2, 1000000]).astype(np.float64)

answer, _, _ = np.histogram2d(*vals, bins=bins, range=ranges)

def test_numpy_perf_2d(benchmark):
    result, _, _ = benchmark(np.histogram2d, *vals, bins=bins, range=ranges)
    assert_array_equal(result, answer)

def make_and_run_hist(hist, axes, vals, fill):
    histo = hist(axes)
    if fill is None:
        histo(*vals)
    elif fill < 0:
        histo.atomic_fill(-fill, *vals)
    else:
        histo.threaded_fill(fill, *vals)

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
def test_2d(benchmark, hist, axis, fill):
    result = benchmark(make_and_run_hist, hist, [axis(bins[0], *ranges[0]), axis(bins[1], *ranges[1])], vals, fill)
    assert_array_equal(result[:-1,:-1], answer[:-1,:-1])

