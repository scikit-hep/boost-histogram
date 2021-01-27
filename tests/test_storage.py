# -*- coding: utf-8 -*-
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import boost_histogram as bh


@pytest.mark.parametrize(
    "storage",
    [bh.storage.Int64, bh.storage.Double, bh.storage.AtomicInt64, bh.storage.Unlimited],
)
def test_setting(storage):
    h = bh.Histogram(bh.axis.Regular(10, 0, 1), storage=storage())

    h[0] = 2
    h[1] = 3
    h[-1] = 5

    assert h[0] == 2
    assert h[1] == 3
    assert h[9] == 5

    assert_array_equal(h.view(), [2, 3, 0, 0, 0, 0, 0, 0, 0, 5])


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
    assert b["variance"][0] == h[0].variance

    h[0] = bh.accumulators.WeightedSum(value=3, variance=1)

    assert h[0].value == 3
    assert h[0].variance == 1

    assert a[0] == h[0]

    assert b["value"][0] == h[0].value
    assert b["variance"][0] == h[0].variance

    assert b[0]["value"] == a[0]["value"]
    assert b[0]["variance"] == a[0]["variance"]

    assert b["value"][0] == a["value"][0]
    assert b["variance"][0] == a["variance"][0]

    assert_array_equal(a.view().value, b.view()["value"])
    assert_array_equal(a.view().variance, b.view()["variance"])


def test_sum_weight():
    h = bh.Histogram(bh.axis.Integer(0, 10), storage=bh.storage.Weight())
    h.fill([1, 2, 3, 3, 3, 4, 5])
    v = h.view().copy()
    res = np.sum(v)
    hres = h.sum()
    assert res.value == hres.value == 7
    assert res.variance == hres.variance == 7

    v2 = v + v
    h2 = h + h

    assert_array_equal(h2.view(), v2)


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
    assert b["count"][0] == h[0].count
    assert b["sum_of_deltas_squared"][0] == h[0].sum_of_deltas_squared

    h[0] = bh.accumulators.Mean(count=6, value=3, variance=2)
    assert h[0].count == 6
    assert h[0].value == 3
    assert h[0].variance == 2

    assert a[0] == h[0]

    assert b["value"][0] == h[0].value
    assert b["count"][0] == h[0].count
    assert b["sum_of_deltas_squared"][0] == h[0].sum_of_deltas_squared

    assert b[0]["value"] == a[0]["value"]
    assert b[0]["count"] == a[0]["count"]
    assert b[0]["sum_of_deltas_squared"] == a[0]["sum_of_deltas_squared"]

    assert b[0]["value"] == a["value"][0]
    assert b[0]["count"] == a["count"][0]
    assert b[0]["sum_of_deltas_squared"] == a["sum_of_deltas_squared"][0]

    assert_array_equal(a.view().value, b.view()["value"])
    assert_array_equal(a.view().count, b.view()["count"])
    assert_array_equal(
        a.view().sum_of_deltas_squared, b.view()["sum_of_deltas_squared"]
    )


def test_setting_weighted_profile():
    h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.WeightedMean())

    h.fill([0.3, 0.3, 0.4, 1.2, 1.6], sample=[1, 2, 3, 4, 4], weight=[1, 1, 1, 1, 2])

    assert h[0] == bh.accumulators.WeightedMean(
        sum_of_weights=3, sum_of_weights_squared=3, value=2, variance=1
    )
    assert h[1] == bh.accumulators.WeightedMean(
        sum_of_weights=3, sum_of_weights_squared=5, value=4, variance=0
    )

    h[0] = bh.accumulators.WeightedMean(
        sum_of_weights=12, sum_of_weights_squared=15, value=11, variance=10
    )
    assert h[0] == bh.accumulators.WeightedMean(
        sum_of_weights=12, sum_of_weights_squared=15, value=11, variance=10
    )

    a = h.view()

    assert a[0] == h[0]

    b = np.asarray(h)

    assert b["value"][0] == h[0].value
    assert b["sum_of_weights"][0] == h[0].sum_of_weights
    assert b["sum_of_weights_squared"][0] == h[0].sum_of_weights_squared
    assert (
        b["_sum_of_weighted_deltas_squared"][0] == h[0]._sum_of_weighted_deltas_squared
    )

    h[0] = bh.accumulators.WeightedMean(
        sum_of_weights=6, sum_of_weights_squared=12, value=3, variance=2
    )

    assert a[0] == h[0]

    assert h[0].value == 3
    assert h[0].variance == 2
    assert h[0].sum_of_weights == 6
    assert h[0].sum_of_weights_squared == 12
    assert h[0]._sum_of_weighted_deltas_squared == 8

    assert b["value"][0] == h[0].value
    assert b["sum_of_weights"][0] == h[0].sum_of_weights
    assert b["sum_of_weights_squared"][0] == h[0].sum_of_weights_squared
    assert (
        b["_sum_of_weighted_deltas_squared"][0] == h[0]._sum_of_weighted_deltas_squared
    )

    assert b[0]["value"] == a[0]["value"]
    assert b[0]["sum_of_weights"] == a[0]["sum_of_weights"]
    assert b[0]["sum_of_weights_squared"] == a[0]["sum_of_weights_squared"]
    assert (
        b[0]["_sum_of_weighted_deltas_squared"]
        == a[0]["_sum_of_weighted_deltas_squared"]
    )

    assert b[0]["value"] == a["value"][0]
    assert b[0]["sum_of_weights"] == a["sum_of_weights"][0]
    assert b[0]["sum_of_weights_squared"] == a["sum_of_weights_squared"][0]
    assert (
        b[0]["_sum_of_weighted_deltas_squared"]
        == a["_sum_of_weighted_deltas_squared"][0]
    )

    assert_array_equal(a.view().value, b.view()["value"])
    assert_array_equal(a.view().sum_of_weights, b.view()["sum_of_weights"])
    assert_array_equal(
        a.view().sum_of_weights_squared, b.view()["sum_of_weights_squared"]
    )
    assert_array_equal(
        a.view()._sum_of_weighted_deltas_squared,
        b.view()["_sum_of_weighted_deltas_squared"],
    )


# Issue #486
def test_modify_weights_by_view():
    bins = [0, 1, 2]
    hist = bh.Histogram(bh.axis.Variable(bins), storage=bh.storage.Weight())
    yields = [3, 4]
    var = [0.1, 0.2]
    hist[...] = np.stack([yields, var], axis=-1)

    hist.view().value /= 2

    assert hist.view().value[0] == pytest.approx(1.5)
    assert hist.view().value[1] == pytest.approx(2)
