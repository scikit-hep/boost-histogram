import pytest

import boost.histogram as bh
from boost.histogram.axis import regular

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose


bins=100
ranges=(-1,1)
bins = np.asarray(bins).astype(np.int64)
ranges = np.asarray(ranges).astype(np.float64)

edges = np.linspace(ranges[0], ranges[1], bins+1)

np.random.seed(42)
vals = np.random.normal(size=[10000000]).astype(np.float32)

answer, _ = np.histogram(vals, bins=bins, range=ranges)

@pytest.mark.benchmark(group='1d-fills')
def test_numpy_perf_1d(benchmark):
    result, _ = benchmark(np.histogram, vals, bins=bins, range=ranges)
    assert_array_equal(result, answer)

def make_and_run_hist(flow, storage):
    histo = bh.histogram(regular(bins, *ranges, flow=flow), storage=storage())
    histo.fill(vals)
    return histo.view()


@pytest.mark.benchmark(group='1d-fills')
@pytest.mark.parametrize("flow", (True, False))
@pytest.mark.parametrize("storage", (bh.storage.int,
                                     bh.storage.double,
                                     bh.storage.unlimited,
                                     # bh.storage.weight,
                                     ))
def test_1d(benchmark, flow, storage):
    result = benchmark(make_and_run_hist, flow, storage)
    assert_allclose(result[:-1], answer[:-1], atol=2)


