import pytest

import numpy as np
from numpy.testing import assert_array_equal

import boost_histogram as bh
from boost_histogram.axis import regular

bins = (100, 100)
ranges = ((-1, 1), (-1, 1))
bins = np.asarray(bins).astype(np.int64)
ranges = np.asarray(ranges).astype(np.float64)

edges = (
    np.linspace(ranges[0, 0], ranges[0, 1], bins[0] + 1),
    np.linspace(ranges[1, 0], ranges[1, 1], bins[1] + 1),
)

np.random.seed(42)
vals = np.random.normal(size=[2, 100000]).astype(np.float64)

answer, _, _ = np.histogram2d(*vals, bins=bins, range=ranges)


@pytest.mark.benchmark(group="2d-fills")
def test_numpy_perf_2d(benchmark):
    result, _, _ = benchmark(np.histogram2d, *vals, bins=bins, range=ranges)
    assert_array_equal(result, answer)


def make_and_run_hist(flow, storage):

    histo = bh.histogram(
        regular(bins[0], *ranges[0], underflow=flow, overflow=flow),
        regular(bins[1], *ranges[1], underflow=flow, overflow=flow),
        storage=storage(),
    )
    histo.fill(*vals)
    return histo.view()


@pytest.mark.benchmark(group="2d-fills")
@pytest.mark.parametrize("flow", (True, False))
@pytest.mark.parametrize(
    "storage",
    (
        bh.storage.int,
        bh.storage.double,
        bh.storage.unlimited,
        # bh.storage.weight,
    ),
)
def test_2d(benchmark, flow, storage):
    result = benchmark(make_and_run_hist, flow, storage)
    assert_array_equal(result[:-1, :-1], answer[:-1, :-1])
