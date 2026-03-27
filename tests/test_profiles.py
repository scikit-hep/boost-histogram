from __future__ import annotations

import numpy as np
from pytest import approx

import boost_histogram as bh


def test_mean_hist():
    h = bh.Histogram(bh.axis.Regular(3, 0, 1), storage=bh.storage.Mean())

    h.fill(0.10, sample=[2.5])
    h.fill(0.25, sample=[3.5])
    h.fill(0.45, sample=[1.2])
    h.fill(0.51, sample=[3.4])
    h.fill(0.81, sample=[1.3])
    h.fill(0.86, sample=[1.9])

    results = (
        {"count": 2, "value": 3.0, "variance": 0.5},
        {"count": 2, "value": 2.3, "variance": 2.42},
        {"count": 2, "value": 1.6, "variance": 0.18},
    )

    for i in range(len(h.axes[0])):
        assert results[i]["count"] == h[i].count
        assert results[i]["value"] == approx(h[i].value)
        assert results[i]["variance"] == approx(h[i].variance)


def test_mean_hist_scalar_axis_broadcast():
    """Test that a scalar axis value is broadcast to match a sample array (issue #727)."""
    data = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [100, 200, 300, 400, 500]])
    h = bh.Histogram(bh.axis.Integer(0, 3), storage=bh.storage.Mean())

    h.fill(0, sample=data[0])
    h.fill(1, sample=data[1])
    h.fill(2, sample=data[2])

    assert h[0].count == 5
    assert h[0].value == approx(3.0)
    assert h[1].count == 5
    assert h[1].value == approx(30.0)
    assert h[2].count == 5
    assert h[2].value == approx(300.0)

    # Verify equivalence with the explicit workaround
    h2 = bh.Histogram(bh.axis.Integer(0, 3), storage=bh.storage.Mean())
    h2.fill([0] * len(data[0]), sample=data[0])
    h2.fill([1] * len(data[1]), sample=data[1])
    h2.fill([2] * len(data[2]), sample=data[2])

    for i in range(3):
        assert h[i].count == h2[i].count
        assert h[i].value == approx(h2[i].value)
        assert h[i].variance == approx(h2[i].variance)


def test_weighted_mean_hist_scalar_axis_broadcast():
    """Test scalar axis broadcast for WeightedMean storage (issue #727)."""
    data = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [100, 200, 300, 400, 500]])
    h = bh.Histogram(bh.axis.Integer(0, 3), storage=bh.storage.WeightedMean())

    h.fill(0, sample=data[0])
    h.fill(1, sample=data[1])
    h.fill(2, sample=data[2])

    assert h[0].sum_of_weights == approx(5.0)
    assert h[0].value == approx(3.0)
    assert h[1].sum_of_weights == approx(5.0)
    assert h[1].value == approx(30.0)
    assert h[2].sum_of_weights == approx(5.0)
    assert h[2].value == approx(300.0)
