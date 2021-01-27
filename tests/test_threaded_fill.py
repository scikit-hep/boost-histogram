# -*- coding: utf-8 -*-
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

import boost_histogram as bh


def fillit(hist, *args, **kwargs):
    return hist.reset().fill(*args, **kwargs)


@pytest.mark.benchmark(group="threaded-fill-1d")
@pytest.mark.parametrize("threads", [1, 4], ids=lambda x: "threads={}".format(x))
# @pytest.mark.parametrize("atomic", [True, False], ids=["atomic", "double"])
@pytest.mark.parametrize(
    "storage", [bh.storage.AtomicInt64, bh.storage.Double, bh.storage.Int64]
)
def test_threads(benchmark, threads, storage):
    axes = [bh.axis.Regular(100, 0, 1)]
    hist_linear = bh.Histogram(*axes, storage=storage())
    hist_atomic = hist_linear.copy()

    vals = np.random.rand(1000000)
    hist_linear.fill(vals)
    hist_result = benchmark(fillit, hist_atomic, vals, threads=threads)

    assert_array_equal(hist_linear, hist_result)


@pytest.mark.parametrize("threads", [1, 4, 7], ids=lambda x: "threads={}".format(x))
@pytest.mark.parametrize(
    "storage", [bh.storage.AtomicInt64, bh.storage.Double, bh.storage.Int64]
)
def test_threaded_builtin(threads, storage):
    axes = [bh.axis.Regular(1000, 0, 1)]
    hist_atomic1 = bh.Histogram(*axes, storage=storage())
    hist_atomic2 = hist_atomic1.copy()

    vals = np.random.rand(10003)

    hist_atomic1.fill(vals)
    hist_atomic2.fill(vals, threads=threads)

    assert_array_equal(hist_atomic1, hist_atomic2)


@pytest.mark.parametrize("threads", [1, 4, 7], ids=lambda x: "threads={}".format(x))
def test_threaded_numpy(threads):
    vals = np.random.rand(10003)

    hist_1, _ = bh.numpy.histogram(vals)
    hist_2, _ = bh.numpy.histogram(vals, threads=threads)

    assert_array_equal(hist_1, hist_2)


@pytest.mark.parametrize("threads", [1, 4, 7], ids=lambda x: "threads={}".format(x))
def test_threaded_weights(threads):
    x, y, weights = np.random.rand(3, 10003)

    hist_1 = bh.Histogram(bh.axis.Regular(10, 0, 1), bh.axis.Regular(10, 0, 1))
    hist_2 = hist_1.copy()

    hist_1.fill(x, y, weight=weights)
    hist_2.fill(x, y, weight=weights, threads=threads)

    assert_almost_equal(hist_1.view(), hist_2.view())


@pytest.mark.parametrize("threads", [1, 4, 7], ids=lambda x: "threads={}".format(x))
def test_threaded_weight_storage(threads):
    x, y, weights = np.random.rand(3, 10003)

    hist_1 = bh.Histogram(
        bh.axis.Regular(10, 0, 1),
        bh.axis.Regular(10, 0, 1),
        storage=bh.storage.Weight(),
    )
    hist_2 = hist_1.copy()

    hist_1.fill(x, y, weight=weights)
    hist_2.fill(x, y, weight=weights, threads=threads)

    assert_almost_equal(hist_1.view().value, hist_2.view().value)
    assert_almost_equal(hist_1.view().variance, hist_2.view().variance)


def test_no_profile():
    hist = bh.Histogram(bh.axis.Regular(10, 0, 1), storage=bh.storage.Mean())
    hist.fill([1, 1], sample=[1, 1])
    with pytest.raises(RuntimeError):
        hist.fill([1, 1], sample=[1, 1], threads=2)


def test_no_weighted_profile():
    hist = bh.Histogram(bh.axis.Regular(10, 0, 1), storage=bh.storage.WeightedMean())
    hist.fill([1, 1], sample=[1, 1])
    with pytest.raises(RuntimeError):
        hist.fill([1, 1], sample=[1, 1], threads=2)


# Filling then summing produces different results for means
# @pytest.mark.parametrize("threads", [2, 4, 7], ids=lambda x: "threads={0}".format(x))
# def test_threaded_samples(threads):
#     x, y, weights = np.random.rand(3, 10003)
#     samples = np.random.randint(1, 10, size=10003)
#
#     hist_1 = bh.Histogram(
#             bh.axis.Regular(10,0,1),
#             bh.axis.Regular(10,0,1),
#             storage=bh.storage.WeightedMean())
#     hist_2 = hist_1.copy()
#
#     hist_1.fill(x, y, sample=samples, weight=weights)
#     hist_2.fill(x, y, sample=samples, weight=weights, threads=threads)
#
#     assert_almost_equal(hist_1.view().value, hist_2.view().value)
#     assert_almost_equal(hist_1.view().variance, hist_2.view().variance)
#     assert_almost_equal(hist_1.view().sum_of_weights, hist_2.view().sum_of_weights)
