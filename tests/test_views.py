# -*- coding: utf-8 -*-
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
