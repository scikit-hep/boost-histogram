# -*- coding: utf-8 -*-
from __future__ import division

import boost_histogram as bh
from numpy.testing import assert_allclose
import pytest


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
