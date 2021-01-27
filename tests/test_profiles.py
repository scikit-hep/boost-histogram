# -*- coding: utf-8 -*-
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
        dict(count=2, value=3.0, variance=0.5),
        dict(count=2, value=2.3, variance=2.42),
        dict(count=2, value=1.6, variance=0.18),
    )

    for i in range(len(h.axes[0])):

        assert results[i]["count"] == h[i].count
        assert results[i]["value"] == approx(h[i].value)
        assert results[i]["variance"] == approx(h[i].variance)
