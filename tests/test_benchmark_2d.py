# -*- coding: utf-8 -*-
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import boost_histogram as bh

STORAGES = (bh.storage.Int64, bh.storage.Double, bh.storage.Unlimited)
DTYPES = (np.float64, np.float32, np.int64, np.int32)

bins = (100, 100)
ranges = ((-1, 1), (-1, 1))
bins = np.asarray(bins).astype(np.int64)
ranges = np.asarray(ranges).astype(np.float64)

edges = (
    np.linspace(ranges[0, 0], ranges[0, 1], bins[0] + 1),
    np.linspace(ranges[1, 0], ranges[1, 1], bins[1] + 1),
)

np.random.seed(42)
vals_core = np.random.normal(size=[2, 100000])
vals = {t: vals_core.astype(t) for t in DTYPES}

answer = {t: np.histogram2d(*vals[t], bins=bins, range=ranges)[0] for t in DTYPES}


@pytest.mark.benchmark(group="2d-fills")
@pytest.mark.parametrize("dtype", vals)
def test_numpy_perf_2d(benchmark, dtype):
    result, _, _ = benchmark(np.histogram2d, *vals[dtype], bins=bins, range=ranges)
    assert_array_equal(result, answer[dtype])


def make_and_run_hist(flow, storage, vals):

    histo = bh.Histogram(
        bh.axis.Regular(bins[0], *ranges[0], underflow=flow, overflow=flow),
        bh.axis.Regular(bins[1], *ranges[1], underflow=flow, overflow=flow),
        storage=storage(),
    )
    histo.fill(*vals)
    return histo.view()


@pytest.mark.benchmark(group="2d-fills")
@pytest.mark.parametrize("dtype", vals)
@pytest.mark.parametrize("storage", STORAGES)
def test_2d(benchmark, flow, storage, dtype):
    result = benchmark(make_and_run_hist, flow, storage, vals[dtype])
    assert_array_equal(result[:-1, :-1], answer[dtype][:-1, :-1])
