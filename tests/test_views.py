# -*- coding: utf-8 -*-
from __future__ import division

import pytest
from numpy.testing import assert_allclose

import boost_histogram as bh


@pytest.fixture
def v():
    h = bh.Histogram(bh.axis.Integer(0, 4), storage=bh.storage.Weight())
    h.fill([1, 1, 1, 2, 2, 3])
    return h.view()


def test_basic_view(v):
    assert_allclose(v.value, [0, 3, 2, 1])
    assert_allclose(v.variance, [0, 3, 2, 1])


def test_view_mul(v):
    v2 = v * 2
    assert_allclose(v2.value, [0, 6, 4, 2])
    assert_allclose(v2.variance, [0, 6, 4, 2])

    v2 = 2 * v
    assert_allclose(v2.value, [0, 6, 4, 2])
    assert_allclose(v2.variance, [0, 6, 4, 2])

    v2 = v * (-2)
    assert_allclose(v2.value, [0, -6, -4, -2])
    assert_allclose(v2.variance, [0, 6, 4, 2])

    v *= 2
    assert_allclose(v.value, [0, 6, 4, 2])
    assert_allclose(v.variance, [0, 6, 4, 2])


def test_view_div(v):
    v2 = v / 2
    assert_allclose(v2.value, [0, 1.5, 1, 0.5])
    assert_allclose(v2.variance, [0, 1.5, 1, 0.5])

    v2 = v / (-0.5)
    assert_allclose(v2.value, [0, -6, -4, -2])
    assert_allclose(v2.variance, [0, 6, 4, 2])

    v2 = 1 / v[1:]
    assert_allclose(v2.value, [1 / 3, 1 / 2, 1])
    assert_allclose(v2.variance, [1 / 3, 1 / 2, 1])

    v /= 0.5
    assert_allclose(v.value, [0, 6, 4, 2])
    assert_allclose(v.variance, [0, 6, 4, 2])


def test_view_add(v):
    v2 = v + 1
    assert_allclose(v2.value, [1, 4, 3, 2])
    assert_allclose(v2.variance, [1, 4, 3, 2])

    v2 = v + 2
    assert_allclose(v2.value, [2, 5, 4, 3])
    assert_allclose(v2.variance, [4, 7, 6, 5])

    v2 = 2 + v
    assert_allclose(v2.value, [2, 5, 4, 3])
    assert_allclose(v2.variance, [4, 7, 6, 5])

    v += 2
    assert_allclose(v.value, [2, 5, 4, 3])
    assert_allclose(v.variance, [4, 7, 6, 5])


def test_view_sub(v):
    v2 = v - 1
    assert_allclose(v2.value, [-1, 2, 1, 0])
    assert_allclose(v2.variance, [1, 4, 3, 2])

    v2 = v - 2
    assert_allclose(v2.value, [-2, 1, 0, -1])
    assert_allclose(v2.variance, [4, 7, 6, 5])

    v2 = 1 - v
    assert_allclose(v2.value, [1, -2, -1, 0])
    assert_allclose(v2.variance, [1, 4, 3, 2])

    v -= 2
    assert_allclose(v.value, [-2, 1, 0, -1])
    assert_allclose(v.variance, [4, 7, 6, 5])


def test_view_unary(v):
    v2 = +v
    assert_allclose(v.value, v2.value)
    assert_allclose(v.variance, v2.variance)

    v2 = -v
    assert_allclose(-v.value, v2.value)
    assert_allclose(v.variance, v2.variance)


def test_view_add_same(v):
    v2 = v + v

    assert_allclose(v.value * 2, v2.value)
    assert_allclose(v.variance * 2, v2.variance)

    v2 = v + v[1]
    assert_allclose(v.value + 3, v2.value)
    assert_allclose(v.variance + 3, v2.variance)

    v2 = v + bh.accumulators.WeightedSum(5, 6)
    assert_allclose(v.value + 5, v2.value)
    assert_allclose(v.variance + 6, v2.variance)

    with pytest.raises(TypeError):
        v2 = v + bh.accumulators.WeightedMean(1, 2, 5, 6)


def test_view_assign(v):
    v[...] = [[4, 1], [5, 2], [6, 1], [7, 2]]

    assert_allclose(v.value, [4, 5, 6, 7])
    assert_allclose(v.variance, [1, 2, 1, 2])


def test_view_assign_mean():
    h = bh.Histogram(bh.axis.Integer(0, 4), storage=bh.storage.Mean())
    m = h.copy().view()

    h[...] = [[10, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    assert_allclose(h.view().count, [10, 4, 7, 10])
    assert_allclose(h.view().value, [2, 5, 8, 11])
    assert_allclose(h.view().variance, [3, 6, 9, 12])

    # Make sure this really was a copy
    assert m.count[0] != 10

    # Assign directly on view
    m[...] = [[10, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

    assert_allclose(m.count, [10, 4, 7, 10])
    assert_allclose(m.value, [2, 5, 8, 11])
    assert_allclose(m.variance, [3, 6, 9, 12])
    # Note: if counts <= 1, variance is undefined


def test_view_assign_wmean():
    h = bh.Histogram(bh.axis.Integer(0, 4), storage=bh.storage.WeightedMean())

    w = h.copy().view()

    h[...] = [[10, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]

    assert_allclose(h.view().sum_of_weights, [10, 5, 9, 13])
    assert_allclose(h.view().sum_of_weights_squared, [2, 6, 10, 14])
    assert_allclose(h.view().value, [3, 7, 11, 15])
    assert_allclose(h.view().variance, [4, 8, 12, 16])

    # Make sure this really was a copy
    assert w.sum_of_weights[0] != 10

    # Assign directly on view
    w[...] = [[10, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]

    assert_allclose(w.sum_of_weights, [10, 5, 9, 13])
    assert_allclose(w.sum_of_weights_squared, [2, 6, 10, 14])
    assert_allclose(w.value, [3, 7, 11, 15])
    assert_allclose(w.variance, [4, 8, 12, 16])
    # Note: if sum_of_weights <= 1, variance is undefined

    w[0] = [9, 1, 2, 3]
    assert w.sum_of_weights[0] == 9
    assert w[0].sum_of_weights_squared == 1
    assert w.value[0] == 2
    assert w[0].variance == 3
