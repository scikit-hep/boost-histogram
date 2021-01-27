# -*- coding: utf-8 -*-
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import boost_histogram as bh


def test_1D_set_bin():

    h = bh.Histogram(bh.axis.Regular(10, 0, 1))

    h[2] = 2
    assert h[2] == 2.0

    h[bh.underflow] = 3
    assert h[bh.underflow] == 3.0

    h[bh.overflow] = 4
    assert h[bh.overflow] == 4.0


def test_2d_set_bin():

    h = bh.Histogram(bh.axis.Regular(10, 0, 1), bh.axis.Regular(10, 0, 1))

    h[2, 4] = 2
    assert h[2, 4] == 2.0

    h[bh.underflow, 5] = 3
    assert h[bh.underflow, 5] == 3.0

    h[bh.overflow, bh.overflow] = 4
    assert h[bh.overflow, bh.overflow] == 4.0


def test_1d_set_array():
    h = bh.Histogram(bh.axis.Regular(10, 0, 1))

    h[...] = np.arange(10)
    assert_array_equal(h.view(), np.arange(10))

    h[...] = np.arange(12)
    assert_array_equal(h.view(flow=True), np.arange(12))

    with pytest.raises(ValueError):
        h[...] = np.arange(9)
    with pytest.raises(ValueError):
        h[...] = np.arange(11)
    with pytest.raises(ValueError):
        h[...] = np.arange(13)

    h[...] = 1
    assert_array_equal(h.view(), np.ones(10))


def test_2d_set_array():
    h = bh.Histogram(bh.axis.Regular(10, 0, 1), bh.axis.Regular(10, 0, 1))

    h[...] = np.arange(10).reshape(-1, 1)
    assert_array_equal(h.view()[:, 2], np.arange(10))

    h[...] = np.arange(12).reshape(-1, 1)
    assert_array_equal(h.view(flow=True)[:, 3], np.arange(12))

    with pytest.raises(ValueError):
        h[...] = np.arange(9).reshape(-1, 1)
    with pytest.raises(ValueError):
        h[...] = np.arange(11).reshape(-1, 1)
    with pytest.raises(ValueError):
        h[...] = np.arange(13).reshape(-1, 1)

    h[...] = 1
    assert_array_equal(h.view(), np.ones((10, 10)))


def test_weighted_set_shortcut():
    h = bh.Histogram(bh.axis.Regular(10, 0, 1), storage=bh.storage.Weight())

    h[0] = (1, 2)

    assert h[0].value == 1
    assert h[0].variance == 2

    h[3:6] = ((3, 4), (5, 6), (7, 8))

    assert h[3].value == 3
    assert h[3].variance == 4
    assert h[4].value == 5
    assert h[4].variance == 6
    assert h[5].value == 7
    assert h[5].variance == 8


@pytest.mark.parametrize(
    "storage, default",
    (
        (bh.storage.Mean, bh.accumulators.Mean(1.0, 2.0, 3.0)),
        (bh.storage.WeightedMean, bh.accumulators.WeightedMean(1.0, 2.0, 3.0, 4.0)),
        (bh.storage.Weight, bh.accumulators.WeightedSum(1.0, 2)),
    ),
)
def test_set_special_dtype(storage, default):
    h = bh.Histogram(
        bh.axis.Regular(10, 0, 1), bh.axis.Regular(10, 0, 1), storage=storage()
    )

    arr = np.full((10, 1), default)
    h[...] = arr
    assert_array_equal(h.view()[:, 1:2], arr)

    arr = np.full((12, 1), default)
    h[...] = arr
    assert_array_equal(h.view(flow=True)[:, 2:3], arr)

    arr = np.full((10, 10), default)
    h[...] = arr
    assert_array_equal(h.view(), arr)

    arr = np.full((10, 12), default)
    h[...] = arr
    assert_array_equal(h.view(flow=True)[1:11, :], arr)

    arr = np.full((12, 10), default)
    h[...] = arr
    assert_array_equal(h.view(flow=True)[:, 1:11], arr)

    arr = np.full((12, 12), default)
    h[...] = arr
    assert_array_equal(h.view(flow=True), arr)

    with pytest.raises(ValueError):
        arr = np.full((9, 1), default)
        h[...] = arr
    with pytest.raises(ValueError):
        arr = np.full((11, 1), default)
        h[...] = arr
    with pytest.raises(ValueError):
        arr = np.full((13, 1), default)
        h[...] = arr

    with pytest.raises(ValueError):
        h[...] = 1

    with pytest.raises(ValueError):
        h[1, 1] = 1
