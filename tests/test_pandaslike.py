# -*- coding: utf-8 -*-
import numpy as np

import boost_histogram as bh


# This is not a reasonable approximation of Pandas, but rather a test for an
# arbitrary convertible object (but if this works, so does Pandas)
class Seriesish(object):
    def __init__(self, array):
        self.array = np.asarray(array)

    def __array__(self):
        return self.array


def test_setting_weighted_profile_convertable():
    h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.WeightedMean())
    data = Seriesish([0.3, 0.3, 0.4, 1.2, 1.6])
    samples = Seriesish([1, 2, 3, 4, 4])
    weights = Seriesish([1, 1, 1, 1, 2])

    h.fill(data, sample=samples, weight=weights)

    assert h[0] == bh.accumulators.WeightedMean(
        sum_of_weights=3, sum_of_weights_squared=3, value=2, variance=1
    )
    assert h[1] == bh.accumulators.WeightedMean(
        sum_of_weights=3, sum_of_weights_squared=5, value=4, variance=0
    )
