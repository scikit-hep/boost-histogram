import pytest
from pytest import approx

import boost.histogram as bh
import numpy as np


def test_weighted_sum():
    v = bh.accumulators.weighted_sum(1.5, 2.5)

    assert v.value == 1.5
    assert v.variance == 2.5

    v += 1.5

    assert v.value == 3.0
    assert v.variance == 4.75

    v = bh.accumulators.weighted_sum()
    v([1,2,3], [4,5,6])

    assert v.value == 6
    assert v.variance == 15



def test_weighted_mean():
    v = bh.accumulators.weighted_mean()
    v(1,4)
    v(2,1)

    assert v.sum_of_weights == 3.0
    assert v.variance == 4.5
    assert v.value == 2.0

    v = bh.accumulators.weighted_mean()
    v([1,2],[4,1])

    assert v.sum_of_weights == 3.0
    assert v.variance == 4.5
    assert v.value == 2.0



def test_mean():
    v = bh.accumulators.mean()
    v(1)
    v(2)
    v(3)

    assert v.count == 3
    assert v.variance == 1
    assert v.value == 2

    v = bh.accumulators.mean()
    v([1,2,3])

    assert v.count == 3
    assert v.variance == 1
    assert v.value == 2


def test_sum():
    v = bh.accumulators.sum()
    v += 1
    v += 2
    v += 3

    assert v.value == 6

    v = bh.accumulators.sum()
    v([1,2,3])
    assert v.value == 6

