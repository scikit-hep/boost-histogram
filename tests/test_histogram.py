# -*- coding: utf-8 -*-
import functools
import operator
import sys
from io import BytesIO

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pytest import approx

import boost_histogram as bh

try:
    import cPickle as pickle
except ImportError:
    import pickle

from collections import OrderedDict

import env


def test_init():
    bh.Histogram()
    bh.Histogram(bh.axis.Integer(-1, 1))
    with pytest.raises(TypeError):
        bh.Histogram(1)
    with pytest.raises(TypeError):
        bh.Histogram("bla")
    with pytest.raises(TypeError):
        bh.Histogram([])
    with pytest.raises(TypeError):
        bh.Histogram(bh.axis.Regular)
    with pytest.raises(TypeError):
        bh.Histogram(bh.axis.Regular())
    with pytest.raises(TypeError):
        bh.Histogram([bh.axis.Integer(-1, 1)])
    with pytest.raises(TypeError):
        bh.Histogram([bh.axis.Integer(-1, 1), bh.axis.Integer(-1, 1)])
    with pytest.raises(TypeError):
        bh.Histogram(bh.axis.Integer(-1, 1), unknown_keyword="nh")

    h = bh.Histogram(bh.axis.Integer(-1, 2))
    assert h.ndim == 1
    assert h.axes[0] == bh.axis.Integer(-1, 2)
    assert h.axes[0].extent == 5
    assert h.axes[0].size == 3
    assert h != bh.Histogram(bh.axis.Regular(1, -1, 1))
    assert h != bh.Histogram(bh.axis.Integer(-1, 1, metadata="ia"))


def test_copy():
    a = bh.Histogram(bh.axis.Integer(-1, 1))
    import copy

    b = copy.copy(a)
    assert a == b
    assert id(a) != id(b)

    c = copy.deepcopy(b)
    assert b == c
    assert id(b) != id(c)

    b = a.copy(deep=False)
    assert a == b
    assert id(a) != id(b)

    c = a.copy()
    assert b == c
    assert id(b) != id(c)


def test_fill_int_1d():

    h = bh.Histogram(bh.axis.Integer(-1, 2))
    assert isinstance(h, bh.Histogram)
    assert h.empty()
    assert h.empty(flow=True)

    with pytest.raises(ValueError):
        h.fill()
    with pytest.raises(ValueError):
        h.fill(1, 2)
    with pytest.raises(TypeError) as k:
        h.fill(1, fiddlesticks=2)
    assert k.value.args[0] == "Keyword(s) fiddlesticks not expected"

    h.fill(-3)
    assert h.empty()
    assert not h.empty(flow=True)
    h.reset()

    for x in (-10, -1, -1, 0, 1, 1, 1, 10):
        h.fill(x)
    assert h.sum() == 6
    assert not h.empty()
    assert not h.empty(flow=True)
    assert h.sum(flow=True) == 8
    assert h.axes[0].extent == 5

    with pytest.raises(IndexError):
        h[0, 1]

    for get in (lambda h, arg: h[arg], lambda h, arg: h[arg]):
        # lambda h, arg: h[arg]):
        assert get(h, 0) == 2
        assert get(h, 1) == 1
        assert get(h, 2) == 3
        # assert get(h, 0).variance == 2
        # assert get(h, 1).variance == 1
        # assert get(h, 2).variance == 3

    assert h[bh.overflow - 1] == 3
    assert h[bh.overflow] == 1
    assert h[bh.underflow] == 1
    assert h[bh.underflow + 1] == 2

    assert h[-1] == 3

    with pytest.raises(IndexError):
        h[3]
    with pytest.raises(IndexError):
        h[-3]


def test_fill_1d(flow):
    h = bh.Histogram(bh.axis.Regular(3, -1, 2, underflow=flow, overflow=flow))
    with pytest.raises(ValueError):
        h.fill()
    with pytest.raises(ValueError):
        h.fill(1, 2)
    for x in (-10, -1, -1, 0, 1, 1, 1, 10):
        h.fill(x)

    assert h.sum() == 6
    assert h.sum(flow=True) == 6 + 2 * flow
    assert h.axes[0].extent == 3 + 2 * flow

    with pytest.raises(IndexError):
        h[0, 1]

    for get in (lambda h, arg: h[arg],):
        # lambda h, arg: h[arg]):
        assert get(h, 0) == 2
        assert get(h, 1) == 1
        assert get(h, 2) == 3
        # assert get(h, 0).variance == 2
        # assert get(h, 1).variance == 1
        # assert get(h, 2).variance == 3

    if flow is True:
        assert get(h, bh.underflow) == 1
        assert get(h, bh.overflow) == 1


@pytest.mark.parametrize(
    "storage",
    [bh.storage.Int64, bh.storage.Double, bh.storage.Unlimited, bh.storage.AtomicInt64],
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


def test_growth():
    h = bh.Histogram(bh.axis.Integer(-1, 2))
    h.fill(-1)
    h.fill(1)
    h.fill(1)
    for _ in range(255):
        h.fill(0)
    h.fill(0)
    for _ in range(1000 - 256):
        h.fill(0)
    print(h.view(flow=True))
    assert h[bh.underflow] == 0
    assert h[0] == 1
    assert h[1] == 1000
    assert h[2] == 2
    assert h[bh.overflow] == 0


def test_growing_cats():
    h = bh.Histogram(
        bh.axis.IntCategory([], growth=True), bh.axis.StrCategory([], growth=True)
    )

    h.fill([1, 2, 1, 1], ["hi", "ho", "hi", "ho"])

    assert h.size == 4


def test_metadata_add():
    h1 = bh.Histogram(
        bh.axis.IntCategory([1, 2, 3]), bh.axis.StrCategory(["1", "2", "3"])
    )
    h2 = bh.Histogram(
        bh.axis.IntCategory([1, 2, 3]), bh.axis.StrCategory(["1", "2", "3"])
    )
    h1.fill([1, 2, 1, 2], ["1", "1", "2", "2"])
    h2.fill([2, 3, 2, 3], ["2", "2", "3", "3"])

    h3 = h1 + h2

    assert h1.axes[0].size == 3
    assert h2.axes[0].size == 3
    assert h3.axes[0].size == 3

    assert h1.axes[1].size == 3
    assert h2.axes[1].size == 3
    assert h3.axes[1].size == 3

    assert h3[bh.loc(2), bh.loc("2")] == 2.0


def test_grow_and_add():
    h1 = bh.Histogram(
        bh.axis.IntCategory([], growth=True), bh.axis.StrCategory([], growth=True)
    )
    h2 = bh.Histogram(
        bh.axis.IntCategory([], growth=True), bh.axis.StrCategory([], growth=True)
    )
    h1.fill([1, 2, 1, 2], ["hi", "hi", "ho", "ho"])
    h2.fill([2, 3, 4, 5], ["ho", "ho", "hum", "hum"])

    h3 = h1 + h2

    assert h1.axes[0].size == 2
    assert h2.axes[0].size == 4
    assert h3.axes[0].size == 5

    assert h1.axes[1].size == 2
    assert h2.axes[1].size == 2
    assert h3.axes[1].size == 3

    assert h3[bh.loc(2), bh.loc("ho")] == 2.0


def test_fill_2d(flow):
    h = bh.Histogram(
        bh.axis.Integer(-1, 2, underflow=flow, overflow=flow),
        bh.axis.Regular(4, -2, 2, underflow=flow, overflow=flow),
    )
    h.fill(-1, -2)
    h.fill(-1, -1)
    h.fill(0, 0)
    h.fill(0, 1)
    h.fill(1, 0)
    h.fill(3, -1)
    h.fill(0, -3)

    with pytest.raises(Exception):
        h.fill(1)
    with pytest.raises(Exception):
        h.fill(1, 2, 3)

    m = [
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    for get in (lambda h, x, y: h[bh.tag.at(x), bh.tag.at(y)],):
        # lambda h, x, y: h[x, y]):
        for i in range(-flow, h.axes[0].size + flow):
            for j in range(-flow, h.axes[1].size + flow):
                assert get(h, i, j) == m[i][j]

    h.fill(1, [1, 2])
    h.fill(np.float64(1), [1, 2])


def test_add_2d(flow):
    h0 = bh.Histogram(
        bh.axis.Integer(-1, 2, underflow=flow, overflow=flow),
        bh.axis.Regular(4, -2, 2, underflow=flow, overflow=flow),
    )
    assert isinstance(h0, bh.Histogram)

    h0.fill(-1, -2)
    h0.fill(-1, -1)
    h0.fill(0, 0)
    h0.fill(0, 1)
    h0.fill(1, 0)
    h0.fill(3, -1)
    h0.fill(0, -3)

    m = [
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    h = h0.copy()
    h += h
    for i in range(-flow, h.axes[0].size + flow):
        for j in range(-flow, h.axes[1].size + flow):
            assert h[bh.tag.at(i), bh.tag.at(j)] == 2 * m[i][j]

    h = sum([h0, h0])
    for i in range(-flow, h.axes[0].size + flow):
        for j in range(-flow, h.axes[1].size + flow):
            assert h[bh.tag.at(i), bh.tag.at(j)] == 2 * m[i][j]

    h = 0 + h0 + h0
    for i in range(-flow, h.axes[0].size + flow):
        for j in range(-flow, h.axes[1].size + flow):
            assert h[bh.tag.at(i), bh.tag.at(j)] == 2 * m[i][j]


@pytest.mark.parametrize("flow", [True, False])
def test_add_2d_fancy(flow):
    h = bh.Histogram(
        bh.axis.Integer(-1, 2, underflow=flow, overflow=flow),
        bh.axis.Regular(4, -2, 2, underflow=flow, overflow=flow),
    )
    assert isinstance(h, bh.Histogram)

    h.fill(-1, -2)
    h.fill(-1, -1)
    h.fill(0, 0)
    h.fill(0, 1)
    h.fill(1, 0)
    h.fill(3, -1)
    h.fill(0, -3)

    m = [
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    h += h

    for i in range(-flow, h.axes[0].size + flow):
        for j in range(-flow, h.axes[1].size + flow):
            assert h[bh.tag.at(i), bh.tag.at(j)] == 2 * m[i][j]


def test_add_2d_bad():
    a = bh.Histogram(bh.axis.Integer(-1, 1))
    b = bh.Histogram(bh.axis.Regular(3, -1, 1))

    with pytest.raises(ValueError):
        a += b


def test_add_2d_w(flow):
    h = bh.Histogram(
        bh.axis.Integer(-1, 2, underflow=flow, overflow=flow),
        bh.axis.Regular(4, -2, 2, underflow=flow, overflow=flow),
    )
    h.fill(-1, -2)
    h.fill(-1, -1)
    h.fill(0, 0)
    h.fill(0, 1)
    h.fill(1, 0)
    h.fill(3, -1)
    h.fill(0, -3)

    m = [
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    h2 = bh.Histogram(
        bh.axis.Integer(-1, 2, underflow=flow, overflow=flow),
        bh.axis.Regular(4, -2, 2, underflow=flow, overflow=flow),
    )
    h2.fill(0, 0, weight=0)

    h2 += h
    h2 += h
    h += h
    assert h == h2

    for i in range(-flow, h.axes[0].size + flow):
        for j in range(-flow, h.axes[1].size + flow):
            assert h[bh.tag.at(i), bh.tag.at(j)] == 2 * m[i][j]


def test_repr():
    hrepr = """Histogram(
  Regular(3, 0, 1),
  Integer(0, 1),
  storage=Double())"""

    h = bh.Histogram(bh.axis.Regular(3, 0, 1), bh.axis.Integer(0, 1))
    assert repr(h) == hrepr

    h.fill([0.3, 0.5], [0, 0])
    hrepr += " # Sum: 2.0"
    assert repr(h) == hrepr

    h.fill([0.3, 12], [3, 0])
    hrepr += " (4.0 with flow)"
    assert repr(h) == hrepr


def test_1d_repr():
    hrepr = """Histogram(Regular(4, 1, 2), storage=Double())"""
    h = bh.Histogram(bh.axis.Regular(4, 1, 2))
    assert repr(h) == hrepr


def test_empty_repr():
    hrepr = """Histogram(storage=Double())"""
    h = bh.Histogram()
    assert repr(h) == hrepr


def test_str():
    h1 = bh.Histogram(bh.axis.Regular(3, 0, 1))
    h1.view(True)[...] = [0, 1, 3, 2, 1]
    assert repr(str(h1)) == repr(
        """                   +---------------------------------------------------------+
[  -inf,      0) 0 |                                                         |
[     0, 0.3333) 1 |===================                                      |
[0.3333, 0.6667) 3 |======================================================== |
[0.6667,      1) 2 |=====================================                    |
[     1,    inf) 1 |===================                                      |
                   +---------------------------------------------------------+"""
    )

    h2 = bh.Histogram(bh.axis.Regular(3, 0, 1), bh.axis.Integer(0, 1))
    assert repr(str(h2)) == repr(repr(h2))


def test_axis():
    axes = (bh.axis.Regular(10, 0, 1), bh.axis.Integer(0, 1))
    h = bh.Histogram(*axes)
    for i, a in enumerate(axes):
        assert h.axes[i] == a
    with pytest.raises(IndexError):
        h.axes[2]
    assert h.axes[-1] == axes[-1]
    assert h.axes[-2] == axes[-2]
    with pytest.raises(IndexError):
        h.axes[-3]


def test_out_of_limit_axis():

    lim = bh._core.hist._axes_limit
    ax = (
        bh.axis.Regular(1, -1, 1, underflow=False, overflow=False) for a in range(lim)
    )
    # Nothrow
    bh.Histogram(*ax)

    ax = (
        bh.axis.Regular(1, -1, 1, underflow=False, overflow=False)
        for a in range(lim + 1)
    )
    with pytest.raises(IndexError):
        bh.Histogram(*ax)


def test_out_of_range():
    h = bh.Histogram(bh.axis.Regular(3, 0, 1))
    h.fill(-1)
    h.fill(2)
    assert h[bh.underflow] == 1
    assert h[bh.overflow] == 1
    with pytest.raises(IndexError):
        h[4]


def test_operators():
    h = bh.Histogram(bh.axis.Integer(0, 2))
    h.fill(0)
    h_orig = h
    h += h
    assert h is h_orig
    assert h[0] == 2
    assert h[1] == 0

    h *= 2
    assert h is h_orig
    assert h[0] == 4
    assert h[1] == 0

    assert (h + h)[0] == (h * 2)[0]
    assert (h + h)[0] == (2 * h)[0]

    h /= 2
    assert h is h_orig
    assert h[0] == 2
    assert h[1] == 0

    assert (h / 2)[0] == 1

    h2 = bh.Histogram(bh.axis.Regular(2, 0, 2))
    with pytest.raises(ValueError):
        h + h2


def test_hist_hist_div():
    h1 = bh.Histogram(bh.axis.Boolean())
    h2 = bh.Histogram(bh.axis.Boolean())

    h1[:] = (8, 6)
    h2[:] = (2, 3)

    h3 = h1 / h2

    assert h3[False] == 4
    assert h3[True] == 2

    h1 /= h2

    assert h1[False] == 4
    assert h1[True] == 2


def test_project():
    h = bh.Histogram(bh.axis.Integer(0, 2), bh.axis.Integer(1, 4))
    h.fill(0, 1)
    h.fill(0, 2)
    h.fill(1, 3)

    h0 = h.project(0)
    assert h0.ndim == 1
    assert h0.axes[0] == bh.axis.Integer(0, 2)
    assert [h0[i] for i in range(2)] == [2, 1]

    h1 = h.project(1)
    assert h1.ndim == 1
    assert h1.axes[0] == bh.axis.Integer(1, 4)
    assert [h1[i] for i in range(3)] == [1, 1, 1]

    with pytest.raises(ValueError):
        h.project(*range(10))

    with pytest.raises(ValueError):
        h.project(2, 1)


def test_shrink_1d():
    h = bh.Histogram(bh.axis.Regular(20, 1, 5))
    h.fill(1.1)
    hs = h[{0: slice(bh.loc(1), bh.loc(2))}]
    assert_array_equal(hs.view(), [1, 0, 0, 0, 0])

    d = OrderedDict({0: slice(bh.loc(1), bh.loc(2))})
    hs = h[d]
    assert_array_equal(hs.view(), [1, 0, 0, 0, 0])


def test_rebin_1d():
    h = bh.Histogram(bh.axis.Regular(20, 1, 5))
    h.fill(1.1)

    hs = h[{0: slice(None, None, bh.rebin(4))}]
    assert_array_equal(hs.view(), [1, 0, 0, 0, 0])

    hs = h[{0: bh.rebin(4)}]
    assert_array_equal(hs.view(), [1, 0, 0, 0, 0])


def test_shrink_rebin_1d():
    h = bh.Histogram(bh.axis.Regular(20, 0, 4))
    h.fill(1.1)
    hs = h[{0: slice(bh.loc(1), bh.loc(3), bh.rebin(2))}]
    assert_array_equal(hs.view(), [1, 0, 0, 0, 0])


def test_rebin_nd():
    h = bh.Histogram(
        bh.axis.Regular(20, 1, 3), bh.axis.Regular(30, 1, 3), bh.axis.Regular(40, 1, 3)
    )

    s = bh.tag.Slicer()

    assert h[{0: s[:: bh.rebin(2)]}].axes.size == (10, 30, 40)
    assert h[{1: s[:: bh.rebin(2)]}].axes.size == (20, 15, 40)
    assert h[{2: s[:: bh.rebin(2)]}].axes.size == (20, 30, 20)

    assert h[{0: s[:: bh.rebin(2)], 2: s[:: bh.rebin(2)]}].axes.size == (10, 30, 20)

    assert h[{1: s[:: bh.sum]}].axes.size == (20, 40)
    assert h[{1: bh.sum}].axes.size == (20, 40)


# CLASSIC: This used to have metadata too, but that does not compare equal
def test_pickle_0():
    a = bh.Histogram(
        bh.axis.IntCategory([0, 1, 2]),
        bh.axis.Integer(0, 20),
        bh.axis.Regular(2, 0.0, 20.0, underflow=False, overflow=False),
        bh.axis.Variable([0.0, 1.0, 2.0]),
        bh.axis.Regular(4, 0, 2 * np.pi, circular=True),
    )
    for i in range(a.axes[0].extent):
        a.fill(i, 0, 0, 0, 0)
        for j in range(a.axes[1].extent):
            a.fill(i, j, 0, 0, 0)
            for k in range(a.axes[2].extent):
                a.fill(i, j, k, 0, 0)
                for l in range(a.axes[3].extent):  # noqa: E741
                    a.fill(i, j, k, l, 0)
                    for m in range(a.axes[4].extent):
                        a.fill(i, j, k, l, m * 0.5 * np.pi)

    io = pickle.dumps(a, -1)
    b = pickle.loads(io)

    assert id(a) != id(b)
    assert a.ndim == b.ndim
    assert a.axes[0] == b.axes[0]
    assert a.axes[1] == b.axes[1]
    assert a.axes[2] == b.axes[2]
    assert a.axes[3] == b.axes[3]
    assert a.axes[4] == b.axes[4]
    assert a.sum() == b.sum()
    assert repr(a) == repr(b)
    assert str(a) == str(b)
    assert a == b


def test_pickle_1():
    a = bh.Histogram(
        bh.axis.IntCategory([0, 1, 2]),
        bh.axis.Integer(0, 3, metadata="ia"),
        bh.axis.Regular(4, 0.0, 4.0, underflow=False, overflow=False),
        bh.axis.Variable([0.0, 1.0, 2.0]),
    )
    assert isinstance(a, bh.Histogram)

    for i in range(a.axes[0].extent):
        a.fill(i, 0, 0, 0, weight=3)
        for j in range(a.axes[1].extent):
            a.fill(i, j, 0, 0, weight=10)
            for k in range(a.axes[2].extent):
                a.fill(i, j, k, 0, weight=2)
                for l in range(a.axes[3].extent):  # noqa: E741
                    a.fill(i, j, k, l, weight=5)

    io = BytesIO()
    pickle.dump(a, io, protocol=-1)
    io.seek(0)
    b = pickle.load(io)

    assert id(a) != id(b)
    assert a.ndim == b.ndim
    assert a.axes[0] == b.axes[0]
    assert a.axes[1] == b.axes[1]
    assert a.axes[2] == b.axes[2]
    assert a.axes[3] == b.axes[3]
    assert a.sum() == b.sum()
    assert repr(a) == repr(b)
    assert str(a) == str(b)
    assert a == b


def test_fill_bool_not_bool():
    h = bh.Histogram(bh.axis.Boolean())

    h.fill([0, 1, 1, 7, -3])

    assert_array_equal(h.view(), [1, 4])


def test_pick_bool():
    h = bh.Histogram(bh.axis.Boolean(), bh.axis.Boolean(metadata={"one": 1}))

    h.fill([True, True, False, False], [True, False, True, True])
    h.fill([True, True, True], True)

    assert_array_equal(h[True, :].view(), [1, 4])
    assert_array_equal(h[False, :].view(), [0, 2])
    assert_array_equal(h[:, False].view(), [0, 1])
    assert_array_equal(h[:, True].view(), [2, 4])


def test_slice_bool():
    h = bh.Histogram(bh.axis.Boolean())
    h.fill([0, 0, 0, 1, 3, 4, -2])

    assert_array_equal(h.view(), [3, 4])
    assert_array_equal(h[1:].view(), [4])
    assert_array_equal(h[:1].view(), [3])

    assert_array_equal(h[:1].axes[0].centers, [0.5])
    assert_array_equal(h[1:].axes[0].centers, [1.5])


def test_pickle_bool():
    a = bh.Histogram(bh.axis.Boolean(), bh.axis.Boolean(metadata={"one": 1}))
    assert isinstance(a, bh.Histogram)

    a.fill([True, True, False, False], [True, False, True, True])
    a.fill([True, True, True], True)

    assert a[True, True] == 4
    assert a[True, False] == 1
    assert a[False, True] == 2
    assert a[False, False] == 0

    io = BytesIO()
    pickle.dump(a, io, protocol=-1)
    io.seek(0)
    b = pickle.load(io)

    assert id(a) != id(b)
    assert a.ndim == b.ndim
    assert a.axes[0] == b.axes[0]
    assert a.axes[1] == b.axes[1]
    assert a.sum() == b.sum()
    assert repr(a) == repr(b)
    assert str(a) == str(b)
    assert a == b
    assert_array_equal(a.view(), b.view())


# Numpy tests


def test_numpy_conversion_0():
    a = bh.Histogram(bh.axis.Integer(0, 3, underflow=False, overflow=False))
    a.fill(0)
    for _ in range(5):
        a.fill(1)
    c = np.array(a)  # a copy
    v = np.asarray(a)  # a view

    for t in (c, v):
        assert t.dtype == np.double  # CLASSIC: np.uint8
        assert_array_equal(t, (1, 5, 0))

    for _ in range(10):
        a.fill(2)
    # copy does not change, but view does
    assert_array_equal(c, (1, 5, 0))
    assert_array_equal(v, (1, 5, 10))

    for _ in range(255):
        a.fill(1)
    c = np.array(a)

    assert c.dtype == np.double  # CLASSIC: np.uint16
    assert_array_equal(c, (1, 260, 10))
    # view does not follow underlying switch in word size
    # assert not np.all(c, v)


def test_numpy_conversion_1():
    # CLASSIC: was weight array
    h = bh.Histogram(bh.axis.Integer(0, 3))
    for _ in range(10):
        h.fill(1, weight=3)
    c = np.array(h)  # a copy
    v = np.asarray(h)  # a view
    assert c.dtype == np.double  # CLASSIC: np.float64
    assert_array_equal(c, np.array((0, 30, 0)))
    assert_array_equal(v, c)


def test_numpy_conversion_2():
    a = bh.Histogram(
        bh.axis.Integer(0, 2, underflow=False, overflow=False),
        bh.axis.Integer(0, 3, underflow=False, overflow=False),
        bh.axis.Integer(0, 4, underflow=False, overflow=False),
    )
    r = np.zeros((2, 3, 4), dtype=np.int8)
    for i in range(a.axes[0].extent):
        for j in range(a.axes[1].extent):
            for k in range(a.axes[2].extent):
                for _ in range(i + j + k):
                    a.fill(i, j, k)
                r[i, j, k] = i + j + k

    d = np.zeros((2, 3, 4), dtype=np.int8)
    for i in range(a.axes[0].extent):
        for j in range(a.axes[1].extent):
            for k in range(a.axes[2].extent):
                d[i, j, k] = a[i, j, k]

    assert_array_equal(d, r)

    c = np.array(a)  # a copy
    v = np.asarray(a)  # a view

    assert_array_equal(c, r)
    assert_array_equal(v, r)


def test_numpy_conversion_3():
    a = bh.Histogram(
        bh.axis.Integer(0, 2),
        bh.axis.Integer(0, 3),
        bh.axis.Integer(0, 4),
        storage=bh.storage.Double(),
    )

    r = np.zeros((4, 5, 6))
    for i in range(a.axes[0].extent):
        for j in range(a.axes[1].extent):
            for k in range(a.axes[2].extent):
                a.fill(i - 1, j - 1, k - 1, weight=i + j + k)
                r[i, j, k] = i + j + k
    c = a.view(flow=True)

    assert_array_equal(c, r)

    assert a.sum() == approx(144)
    assert a.sum(flow=True) == approx(720)
    assert c.sum() == approx(720)


def test_numpy_conversion_4():
    a = bh.Histogram(
        bh.axis.Integer(0, 2, underflow=False, overflow=False),
        bh.axis.Integer(0, 4, underflow=False, overflow=False),
    )
    a1 = np.asarray(a)
    assert a1.dtype == np.double
    assert a1.shape == (2, 4)

    b = bh.Histogram()
    b1 = np.asarray(b)
    assert b1.shape == ()
    assert np.sum(b1) == 0

    # Compare sum methods
    assert b.sum() == np.asarray(b).sum()


def test_numpy_conversion_5():
    a = bh.Histogram(
        bh.axis.Integer(0, 3, underflow=False, overflow=False),
        bh.axis.Integer(0, 2, underflow=False, overflow=False),
        storage=bh.storage.Unlimited(),
    )

    a.fill(0, 0)
    for _ in range(80):
        a = a + a
    # a now holds a multiprecision type
    a.fill(1, 0)
    for _ in range(2):
        a.fill(2, 0)
    for _ in range(3):
        a.fill(0, 1)
    for _ in range(4):
        a.fill(1, 1)
    for _ in range(5):
        a.fill(2, 1)
    a1 = a.view()
    assert a1.shape == (3, 2)
    assert a1[0, 0] == float(2 ** 80)
    assert a1[1, 0] == 1
    assert a1[2, 0] == 2
    assert a1[0, 1] == 3
    assert a1[1, 1] == 4
    assert a1[2, 1] == 5


def test_fill_with_sequence_0():
    def fa(*args):
        return np.array(args, dtype=float)

    def ia(*args):
        return np.array(args, dtype=int)

    a = bh.Histogram(bh.axis.Integer(0, 2))
    a.fill(np.array(1))  # 0-dim arrays work
    a.fill(ia(-1, 0, 1, 2))
    a.fill((2, 1, 0, -1))
    assert_array_equal(a.view(True), [2, 2, 3, 2])

    with pytest.raises(ValueError):
        a.fill(np.empty((2, 2)))
    with pytest.raises(ValueError):
        a.fill(np.empty(2), 1)
    with pytest.raises(ValueError):
        a.fill(np.empty(2), np.empty(3))
    with pytest.raises(ValueError):
        a.fill("abc")

    with pytest.raises(IndexError):
        a[1, 2]

    b = bh.Histogram(bh.axis.Regular(3, 0, 3))
    b.fill(fa(0, 0, 1, 2))
    b.fill(ia(1, 0, 2, 2))
    assert_array_equal(b.view(True), [0, 3, 2, 3, 0])

    c = bh.Histogram(
        bh.axis.Integer(0, 2, underflow=False, overflow=False), bh.axis.Regular(2, 0, 2)
    )
    c.fill(ia(-1, 0, 1), fa(-1.0, 1.5, 0.5))
    assert_array_equal(c.view(True), [[0, 0, 1, 0], [0, 1, 0, 0]])
    # we don't support: assert a[[1, 1]].value, 0

    with pytest.raises(ValueError):
        c.fill(1)
    with pytest.raises(ValueError):
        c.fill([1, 0, 2], [1, 1])

    # this broadcasts
    c.fill([1, 0], -1)
    assert_array_equal(c.view(True), [[1, 0, 1, 0], [1, 1, 0, 0]])
    c.fill([1, 0], 0)
    assert_array_equal(c.view(True), [[1, 1, 1, 0], [1, 2, 0, 0]])
    c.fill(0, [-1, 0.5, 1.5, 2.5])
    assert_array_equal(c.view(True), [[2, 2, 2, 1], [1, 2, 0, 0]])

    with pytest.raises(IndexError):
        c[1]
    with pytest.raises(IndexError):
        c[1, 2, 3]


def test_fill_with_sequence_1():
    def fa(*args):
        return np.array(args, dtype=float)

    a = bh.Histogram(bh.axis.Integer(0, 3), storage=bh.storage.Weight())
    v = fa(-1, 0, 1, 2, 3, 4)
    w = fa(2, 3, 4, 5, 6, 7)  # noqa
    a.fill(v, weight=w)
    a.fill((0, 1), weight=(2, 3))

    assert a[bh.underflow] == bh.accumulators.WeightedSum(2, 4)
    assert a[0] == bh.accumulators.WeightedSum(5, 13)
    assert a[1] == bh.accumulators.WeightedSum(7, 25)
    assert a[2] == bh.accumulators.WeightedSum(5, 25)

    assert a[bh.underflow].value == 2
    assert a[0].value == 5
    assert a[1].value == 7
    assert a[2].value == 5

    assert a[bh.underflow].variance == 4
    assert a[0].variance == 13
    assert a[1].variance == 25
    assert a[2].variance == 25

    a.fill((1, 2), weight=1)
    a.fill(0, weight=1)
    a.fill(0, weight=2)
    assert a[0].value == 8
    assert a[1].value == 8
    assert a[2].value == 6

    with pytest.raises(TypeError):
        a.fill((1, 2), foo=(1, 1))
    with pytest.raises(ValueError):
        a.fill((1, 2, 3), weight=(1, 2))
    with pytest.raises(ValueError):
        a.fill((1, 2), weight="ab")
    with pytest.raises(TypeError):
        a.fill((1, 2), weight=(1, 1), foo=1)
    with pytest.raises(ValueError):
        a.fill((1, 2), weight=([1, 1], [2, 2]))

    a = bh.Histogram(bh.axis.Integer(0, 3))
    # this broadcasts
    a.fill((1, 2), weight=1)
    assert a[1] == 1.0
    assert a[2] == 1.0

    a = bh.Histogram(
        bh.axis.Integer(0, 2, underflow=False, overflow=False),
        bh.axis.Regular(2, 0, 2, underflow=False, overflow=False),
    )
    a.fill((-1, 0, 1), (-1, 1, 0.1))
    assert a[0, 0] == 0
    assert a[0, 1] == 1
    assert a[1, 0] == 1
    assert a[1, 1] == 0
    a = bh.Histogram(bh.axis.Integer(0, 3, underflow=False, overflow=False))
    a.fill((0, 0, 1, 2))
    a.fill((1, 0, 2, 2))
    assert a[0] == 3
    assert a[1] == 2
    assert a[2] == 3


def test_fill_with_sequence_2():
    a = bh.Histogram(bh.axis.StrCategory(["A", "B"]))
    a.fill("A")
    a.fill(np.array("B"))  # 0-dim array is also accepted
    a.fill(("A", "B", "C"))
    assert_array_equal(a.view(True), [2, 2, 1])
    a.fill(np.array(("D", "B", "A"), dtype="S5"))
    a.fill(np.array(("D", "B", "A"), dtype="U1"))
    assert_array_equal(a.view(True), [4, 4, 3])

    with pytest.raises(ValueError):
        a.fill(np.array((("B", "A"), ("C", "A"))))  # ndim == 2 not allowed

    b = bh.Histogram(
        bh.axis.Integer(0, 2, underflow=False, overflow=False),
        bh.axis.StrCategory(["A", "B"]),
    )
    b.fill((1, 0, 10), ("C", "B", "A"))
    assert_array_equal(b.view(True), [[0, 1, 0], [0, 0, 1]])


def test_fill_with_sequence_3():
    h = bh.Histogram(bh.axis.StrCategory([], growth=True))
    h.fill("A")
    assert h.axes[0].size == 1
    h.fill(["A"])
    assert h.axes[0].size == 1
    h.fill(["A", "B"])
    assert h.axes[0].size == 2
    assert_array_equal(h.view(True), [3, 1])


def test_fill_with_sequence_4():
    h = bh.Histogram(
        bh.axis.StrCategory([], growth=True), bh.axis.Integer(0, 0, growth=True)
    )
    h.fill("1", np.arange(2))
    assert h.axes[0].size == 1
    assert h.axes[1].size == 2
    assert_array_equal(h.view(True), [[1, 1]])

    with pytest.raises(ValueError):
        h.fill(["1"], np.arange(2))  # lengths do not match


def test_axes_reference():
    h = bh.Histogram(
        bh.axis.Regular(10, 0, 1),
        bh.axis.Regular(20, 2, 4, metadata=12),
        bh.axis.StrCategory([], growth=True),
    )

    h.axes[0].metadata = "set1"
    h.axes[1].metadata = None

    h_copy = h[...]

    assert h_copy.axes[0].metadata == "set1"
    assert h_copy.axes[1].metadata is None

    assert h_copy.axes[2].size == 0

    h_copy.fill([0.3], [3.2], ["check"])

    assert h_copy.axes[2].size == 1


def test_axes_lifetime():
    h = bh.Histogram(bh.axis.Regular(10, 0, 1, metadata=2))

    ax = h.axes[0]

    if env.CPYTHON:
        # 2 is the minimum refcount, so the *python* object should be deleted
        # after the del; hopefully the C++ object lives through the axis instance.
        assert sys.getrefcount(h) == 2

    del h

    assert ax.metadata == 2
    ax.metadata = 3
    assert ax.metadata == 3


def test_copy_axes():
    h = bh.Histogram(bh.axis.Regular(10, 0, 1))

    h2 = h.copy()

    h.axes[0].metadata = 1
    assert h2.axes[0].metadata is None


def test_shape():
    h = bh.Histogram(
        bh.axis.Regular(7, 0, 1),
        bh.axis.Regular(13, 0, 1),
        bh.axis.Regular(17, 0, 1),
        bh.axis.Regular(24, 0, 1),
    )

    assert h.shape == (7, 13, 17, 24)


def test_empty_shape():
    h = bh.Histogram()
    assert h.shape == ()


# issue #416 a
def test_hist_division():
    edges = [0, 0.25, 0.5, 0.75, 1, 2, 3, 4, 7, 10]
    edges = [-x for x in reversed(edges)] + edges[1:]

    h = bh.Histogram(bh.axis.Variable(edges))
    h[...] = 1
    h1 = h.copy()

    dens = h.view().copy()
    dens /= h.axes[0].widths * h.sum()

    h1 /= h.axes[0].widths * h.sum()

    assert_array_equal(h1.view(), dens)


# issue #416 b
# def test_hist_division():
#     edges = [0, .25, .5, .75, 1, 2, 3, 4, 7, 10]
#    edges = [-x for x in reversed(edges)] + edges[1:]
#
#    h = bh.Histogram(bh.axis.Variable(edges))
#    h[...] = 1
#
#    dens = h.view().copy() / h.axes[0].widths * h.sum()
#    h1 = h.copy()
#
#    h1[:] /=  h.axes[0].widths * h.sum()
#
#    assert_allclose(h1.view(), dens)


def test_add_hists():
    edges = [0, 0.25, 0.5, 0.75, 1, 2, 3, 4, 7, 10]
    edges = [-x for x in reversed(edges)] + edges[1:]

    h = bh.Histogram(bh.axis.Variable(edges))
    h[...] = 1

    h1 = h.copy()
    h1 += h.view()

    h2 = h.copy()
    h2 += h1

    h3 = h.copy()
    h3 += 5

    assert_array_equal(h, 1)
    assert_array_equal(h1, 2)
    assert_array_equal(h2, 3)
    assert_array_equal(h3, 6)


def test_add_broadcast():
    h = bh.Histogram(bh.axis.Regular(10, 0, 1), bh.axis.Regular(20, 0, 1))

    h1 = h.copy()
    h2 = h.copy()

    h1[...] = 1
    assert h1.view().sum() == 10 * 20
    assert h1.view(flow=True).sum() == 10 * 20

    h2 = h + [[1]]
    assert h2.sum() == 10 * 20
    assert h2.sum(flow=True) == 10 * 20

    h3 = h + np.ones((10, 20))
    assert h3.sum() == 10 * 20
    assert h3.sum(flow=True) == 10 * 20

    h4 = h + np.ones((12, 22))
    assert h4.view(flow=True).sum() == 12 * 22

    h5 = h + np.ones((10, 1))
    assert h5.sum(flow=True) == 10 * 20

    h5 = h + np.ones((1, 22))
    assert h5.sum(flow=True) == 12 * 22


# Issue #431
def test_mul_shallow():
    import threading

    my_lock = threading.Lock()

    h = bh.Histogram(bh.axis.Integer(0, 3, metadata=my_lock), metadata=my_lock)
    h.fill([0, 0, 0, 1])

    h2 = h * 2

    assert h.metadata is h2.metadata
    assert h.axes[0].metadata is h2.axes[0].metadata


def test_reductions():
    h = bh.Histogram(bh.axis.Variable([1, 2, 4, 7, 9, 9.5, 10]))

    widths_1 = functools.reduce(operator.mul, h.axes.widths)
    widths_2 = np.prod(h.axes.widths, axis=0)

    assert_array_equal(widths_1, widths_2)


# Issue 435
def test_np_scalars():
    hist = bh.Histogram(bh.axis.Regular(30, 1, 500, transform=bh.axis.transform.log))
    hist.fill([7, 7])

    hist2 = hist / np.float64(2.0)
    assert hist2[bh.loc(7)] == 1.0

    hist2 = hist / hist.axes.widths.prod(axis=0)
    assert hist2[bh.loc(7)] == approx(1.3467513416439476)

    with pytest.raises(ValueError):
        hist / np.array([1, 2, 3])

    hist /= np.float64(2.0)
    assert hist[bh.loc(7)] == 1.0
