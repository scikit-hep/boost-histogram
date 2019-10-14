import pytest
from pytest import approx

import boost_histogram as bh
import numpy as np


def test_weighted_sum():
    a = bh.accumulators.weighted_sum(1.5, 2.5)

    assert a.value == 1.5
    assert a.variance == 2.5

    a += 1.5

    assert a.value == 3.0
    assert a.variance == 4.75

    vals = [1, 2, 3]
    vari = [4, 5, 6]

    a = bh.accumulators.weighted_sum()
    for val, var in zip(vals, vari):
        a += bh.accumulators.weighted_sum(val, variance=var)

    assert a.value == 6
    assert a.variance == 15

    a2 = bh.accumulators.weighted_sum().fill(vals, variance=vari)
    assert a == a2

    assert a == bh.accumulators.weighted_sum(6, 15)


def test_sum():
    vals = [1, 2, 3]
    a = bh.accumulators.sum()
    for val in vals:
        a += val

    assert a.value == 6

    a2 = bh.accumulators.sum().fill(vals)
    assert a == a2

    assert a == bh.accumulators.sum(6)


def test_weighted_mean():
    vals = [4, 1]
    weights = [1, 2]
    a = bh.accumulators.weighted_mean()
    for v, w in zip(vals, weights):
        a(v, weight=w)

    assert a.sum_of_weights == 3.0
    assert a.variance == 4.5
    assert a.value == 2.0

    a2 = bh.accumulators.weighted_mean().fill(vals, weight=weights)
    assert a == a2

    assert a == bh.accumulators.weighted_mean(3, 5, 2, 4.5)


def test_mean():
    vals = [1, 2, 3]
    a = bh.accumulators.mean()
    for val in vals:
        a(val)

    assert a.count == 3
    assert a.value == 2
    assert a.variance == 1

    a2 = bh.accumulators.mean().fill([1, 2, 3])
    assert a == a2

    assert a == bh.accumulators.mean(3, 2, 1)
