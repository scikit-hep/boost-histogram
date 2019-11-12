import pytest
import boost_histogram as bh
import numpy as np
from numpy.testing import assert_array_equal


@pytest.mark.parametrize(
    "storage",
    [bh.storage.Int, bh.storage.Double, bh.storage.AtomicInt, bh.storage.Unlimited],
)
def test_setting(storage):
    h = bh.Histogram(bh.axis.Regular(10, 0, 1), storage=storage())
    h[bh.underflow] = 1
    h[0] = 2
    h[1] = 3
    h[bh.loc(0.55)] = 4
    h[-1] = 5
    h[bh.overflow] = 6

    assert h[bh.underflow] == 1
    assert h[0] == 2
    assert h[1] == 3
    assert h[bh.loc(0.55)] == 4
    assert h[5] == 4
    assert h[-1] == 5
    assert h[9] == 5
    assert h[bh.overflow] == 6

    assert_array_equal(h.view(flow=True), [1, 2, 3, 0, 0, 0, 4, 0, 0, 0, 5, 6])


def test_setting_weight():
    h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.Weight())

    h.fill([0.3, 0.3, 0.4, 1.2])

    assert h[0] == bh.accumulators.WeightedSum(3, 3)
    assert h[1] == bh.accumulators.WeightedSum(1, 1)

    h[0] = bh.accumulators.WeightedSum(value=2, variance=2)
    assert h[0] == bh.accumulators.WeightedSum(2, 2)

    a = h.view()

    assert a[0] == h[0]

    b = np.asarray(h)
    assert b["value"][0] == h[0].value

    h[0] = bh.accumulators.WeightedSum(value=3, variance=1)

    assert a[0] == h[0]
    assert b["value"][0] == h[0].value


def test_setting_profile():
    h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.Mean())

    h.fill([0.3, 0.3, 0.4, 1.2, 1.6], sample=[1, 2, 3, 4, 4])

    assert h[0] == bh.accumulators.Mean(count=3, value=2, variance=1)
    assert h[1] == bh.accumulators.Mean(count=2, value=4, variance=0)

    h[0] = bh.accumulators.Mean(count=12, value=11, variance=10)
    assert h[0] == bh.accumulators.Mean(count=12, value=11, variance=10)

    a = h.view()

    assert a[0] == h[0]

    b = np.asarray(h)
    assert b["value"][0] == h[0].value

    h[0] = bh.accumulators.Mean(count=6, value=3, variance=2)

    assert a[0] == h[0]
    assert b["value"][0] == h[0].value


def test_setting_weighted_profile():
    h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.WeightedMean())

    h.fill([0.3, 0.3, 0.4, 1.2, 1.6], sample=[1, 2, 3, 4, 4], weight=[1, 1, 1, 1, 2])

    assert h[0] == bh.accumulators.WeightedMean(wsum=3, wsum2=3, value=2, variance=1)
    assert h[1] == bh.accumulators.WeightedMean(wsum=3, wsum2=5, value=4, variance=0)

    h[0] = bh.accumulators.WeightedMean(wsum=12, wsum2=15, value=11, variance=10)
    assert h[0] == bh.accumulators.WeightedMean(
        wsum=12, wsum2=15, value=11, variance=10
    )

    a = h.view()

    assert a[0] == h[0]

    b = np.asarray(h)
    assert b["value"][0] == h[0].value

    h[0] = bh.accumulators.WeightedMean(wsum=6, wsum2=12, value=3, variance=2)

    assert a[0] == h[0]
    assert b["value"][0] == h[0].value
