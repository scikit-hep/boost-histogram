from __future__ import annotations

import platform

import numpy as np
import pytest
from pytest import approx

import boost_histogram as bh

STORAGES = (bh.storage.Int64, bh.storage.Double, bh.storage.Unlimited)
DTYPES = (np.float64, np.float32, np.int64, np.int32)

# Casting is broken for numpy on ppc64le: https://github.com/numpy/numpy/issues/21062
if platform.machine() == "ppc64le":
    DTYPES = (np.float64, np.int64)

bins = 100
ranges = (-1, 1)
bins = np.asarray(bins).astype(np.int64)
ranges = np.asarray(ranges).astype(np.float64)

edges = np.linspace(ranges[0], ranges[1], bins + 1)

np.random.seed(42)
vals_core = np.random.normal(size=[100000])
vals = {t: vals_core.astype(t) for t in DTYPES}

answer = {t: np.histogram(vals[t], bins=bins, range=ranges)[0] for t in DTYPES}


@pytest.mark.benchmark(group="1d-fills")
@pytest.mark.parametrize("dtype", vals)
def test_numpy_1d(benchmark, dtype):
    result, _ = benchmark(np.histogram, vals[dtype], bins=bins, range=ranges)
    assert result == approx(answer[dtype])


def make_and_run_hist(flow, storage, vals):
    histo = bh.Histogram(
        bh.axis.Regular(bins, *ranges, underflow=flow, overflow=flow), storage=storage()
    )
    histo.fill(vals)
    return histo.view()


@pytest.mark.benchmark(group="1d-fills")
@pytest.mark.parametrize("dtype", vals)
@pytest.mark.parametrize("storage", STORAGES)
def test_boost_1d(benchmark, flow, storage, dtype):
    result = benchmark(make_and_run_hist, flow, storage, vals[dtype])
    assert result[:-1] == approx(answer[dtype][:-1], atol=2)
