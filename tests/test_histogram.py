from __future__ import annotations

import copy
import functools
import operator
import pickle
import platform
import subprocess
import sys
import threading
from collections import OrderedDict
from io import BytesIO

import numpy as np
import pytest
from pytest import approx

import boost_histogram as bh


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
    with pytest.raises(TypeError):
        h.fill(1, fiddlesticks=2)

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
    assert h[-3] == h[0]

    with pytest.raises(IndexError):
        h[3]
    with pytest.raises(IndexError):
        h[-4]


def test_fill_int_with_float_single_1d():
    h = bh.Histogram(bh.axis.Integer(-1, 2))
    with pytest.raises(TypeError):
        h.fill(0.3)


def test_fill_int_with_float_array_1d():
    h = bh.Histogram(bh.axis.Integer(-1, 2))
    with pytest.raises(TypeError):
        h.fill([-0.3, 0.3])


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


def test_setting(count_single_storage):
    h = bh.Histogram(bh.axis.Regular(10, 0, 1), storage=count_single_storage())
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

    assert h.view(flow=True) == approx([1, 2, 3, 0, 0, 0, 4, 0, 0, 0, 5, 6])


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


def test_noflow_cats():
    h = bh.Histogram(
        bh.axis.IntCategory([1, 2, 3], overflow=False),
        bh.axis.StrCategory(["hi"], overflow=False),
    )

    h.fill([1, 2, 3, 4], ["hi", "ho", "hi", "ho"])

    assert h.sum() == 2


def _has_issue_960_bug():
    # Probe for the upstream Boost.Histogram bug where a broadcast scalar argument
    # invalidates an entire bulk fill when an earlier non-inclusive axis dropped the
    # first array entry. Returns True while the (vendored) bug is present.
    h = bh.Histogram(bh.axis.IntCategory([1], overflow=False), bh.axis.IntCategory([1]))
    h.fill([0, 1], 1)  # first entry (0) is out of range on the non-inclusive axis
    return h.sum() == 0


@pytest.mark.xfail(
    _has_issue_960_bug(),
    reason="upstream Boost.Histogram bug, fixed upstream; pending boost bump (#960)",
    strict=True,
)
def test_noflow_cat_scalar_broadcast():
    # https://github.com/scikit-hep/boost-histogram/issues/960
    # A broadcast scalar on a later axis must not zero the whole fill when an earlier
    # non-inclusive axis drops the first array entry.
    h = bh.Histogram(
        bh.axis.IntCategory([1, 2, 3], overflow=False),
        bh.axis.StrCategory(["nominal"]),
        storage=bh.storage.Weight(),
    )
    values = np.array([-1, 1, 2, 3, 1])  # first entry (-1) is out of range
    h.fill(values, "nominal", weight=np.ones_like(values, dtype=float))

    # only the out-of-range -1 is dropped; everything else is kept
    assert h.sum().value == 4
    assert h[bh.loc(1), bh.loc("nominal")].value == 2
    assert h[bh.loc(2), bh.loc("nominal")].value == 1
    assert h[bh.loc(3), bh.loc("nominal")].value == 1


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

    # TODO: this really should be a TypeError, but we are throwing ValueError
    with pytest.raises(ValueError):
        h.fill(1)
    with pytest.raises(ValueError):
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


def test_sub_2d(flow, count_storage):
    h0 = bh.Histogram(
        bh.axis.Integer(-1, 2, underflow=flow, overflow=flow),
        bh.axis.Regular(4, -2, 2, underflow=flow, overflow=flow),
        storage=count_storage(),
    )

    h0.fill(-1, -2)
    h0.fill(-1, -1)
    h0.fill(0, 0)
    h0.fill(0, 1)
    h0.fill(1, 0)
    h0.fill(3, -1)
    h0.fill(0, -3)

    m = h0.values(flow=True).copy()

    if count_storage not in {bh.storage.AtomicInt64, bh.storage.Weight}:
        h = h0.copy()
        h -= h0
        assert h.values(flow=True) == approx(m * 0)

        h -= h0
        assert h.values(flow=True) == approx(-m)

        h2 = h0 - (h0 + h0 + h0)
        assert h2.values(flow=True) == approx(-2 * m)

    h3 = h0 - h0.view(flow=True) * 4
    assert h3.values(flow=True) == approx(-3 * m)

    h4 = h0.copy()
    h4 -= h0.view(flow=True) * 5
    assert h4.values(flow=True) == approx(-4 * m)

    h5 = h0.copy()
    h5 -= 2
    assert h5.values(flow=True) == approx(m - 2)

    h6 = h0 - 3
    assert h6.values(flow=True) == approx(m - 3)


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


def test_empty_mean_repr_has_no_sum_line():
    # Mean/WeightedMean accumulators have no __bool__, so an empty sum is
    # still truthy; the repr must not print a spurious "# Sum:" for it.
    h = bh.Histogram(bh.axis.Regular(3, 0, 1), storage=bh.storage.Mean())
    assert "# Sum:" not in repr(h)

    h2 = bh.Histogram(bh.axis.Regular(3, 0, 1), storage=bh.storage.WeightedMean())
    assert "# Sum:" not in repr(h2)

    h.fill([0.5], sample=[1.0])
    assert "# Sum:" in repr(h)


def test_str():
    h1 = bh.Histogram(bh.axis.Regular(3, 0, 1))
    h1.view(True)[...] = [0, 1, 3, 2, 1]
    current_repr = repr(str(h1))

    assert "[  -inf,      0)" in current_repr
    assert "[0.3333, 0.6667) 3" in current_repr
    assert "[     1,    inf)" in current_repr

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


@pytest.mark.parametrize("operation", [operator.add, operator.iadd])
@pytest.mark.parametrize(
    ("left_storage", "right_storage"),
    [
        (bh.storage.Int64, bh.storage.Double),
        (bh.storage.Double, bh.storage.Int64),
    ],
)
def test_add_mismatched_storage_raises(operation, left_storage, right_storage):
    left = bh.Histogram(bh.axis.Regular(3, -1, 1), storage=left_storage())
    right = bh.Histogram(bh.axis.Regular(3, -1, 1), storage=right_storage())
    right.fill(0)

    message = (
        f"different storage types: {left_storage.__name__} and {right_storage.__name__}"
    )
    with pytest.raises(TypeError, match=message):
        operation(left, right)


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


@pytest.mark.parametrize("storage", [bh.storage.Int64, bh.storage.AtomicInt64])
def test_int_storage_division_promotes_to_double(storage):
    # gh #601: dividing integer-storage histograms must not truncate.
    a = bh.Histogram(bh.axis.Integer(0, 3), storage=storage())
    b = bh.Histogram(bh.axis.Integer(0, 3), storage=storage())
    a[:] = (5, 3, 7)
    b[:] = (2, 4, 3)

    expected = np.array([5, 3, 7]) / np.array([2, 4, 3])

    # hist / hist
    result = a / b
    assert result.storage_type is bh.storage.Double
    assert result.view() == approx(expected)
    # operands are unchanged
    assert a.storage_type is storage
    assert b.storage_type is storage

    # hist / scalar
    assert (a / 2).view() == approx(np.array([5, 3, 7]) / 2)
    assert (a / 2).storage_type is bh.storage.Double

    # in-place division promotes storage but keeps the same object
    a_orig = a
    a /= b
    assert a is a_orig
    assert a.storage_type is bh.storage.Double
    assert a.view() == approx(expected)


@pytest.mark.parametrize("storage", [bh.storage.Int64, bh.storage.AtomicInt64])
def test_int_storage_mul_by_float_promotes_to_double(storage):
    # A non-integer scalar multiply must promote like division, instead of
    # leaking a numpy UFuncTypeError from the in-place int64 view multiply.
    a = bh.Histogram(bh.axis.Integer(0, 3), storage=storage())
    a[:] = (1, 2, 3)

    result = a * 0.5
    assert result.storage_type is bh.storage.Double
    assert result.view() == approx(np.array([0.5, 1.0, 1.5]))
    # operand is unchanged
    assert a.storage_type is storage

    # multiplying by an integer scalar keeps the int storage
    int_result = a * 2
    assert int_result.storage_type is storage
    assert int_result.view() == approx(np.array([2, 4, 6]))

    # in-place multiply promotes storage but keeps the same object
    a_orig = a
    a *= 0.5
    assert a is a_orig
    assert a.storage_type is bh.storage.Double
    assert a.view() == approx(np.array([0.5, 1.0, 1.5]))


def test_mixed_int_double_division():
    # gh #601: storage mismatch must still divide correctly.
    ai = bh.Histogram(bh.axis.Integer(0, 3), storage=bh.storage.Int64())
    bd = bh.Histogram(bh.axis.Integer(0, 3), storage=bh.storage.Double())
    ai[:] = (5, 3, 7)
    bd[:] = (2, 4, 3)

    expected = np.array([5, 3, 7]) / np.array([2, 4, 3])
    assert (ai / bd).view() == approx(expected)
    assert (bd / ai).view() == approx(1 / expected)


def test_reflected_scalar_operators():
    # gh-1154: scalar - hist and scalar / hist were missing (only the
    # commutative __radd__/__rmul__ existed).
    h = bh.Histogram(bh.axis.Regular(4, 0, 4))
    h[:] = (1, 2, 4, 5)
    values = h.view().copy()

    # scalar - hist == -(hist - scalar)
    assert (10 - h).view() == approx(10 - values)
    assert (10 - h).view() == approx(((h - 10) * -1).view())
    # scalar / hist, elementwise
    assert (8 / h).view() == approx(8 / values)
    # operands left unchanged
    assert h.view() == approx(values)


def test_reflected_division_promotes_int_storage():
    # gh-1154: scalar / int-hist must not truncate (mirrors __itruediv__).
    h = bh.Histogram(bh.axis.Integer(0, 3), storage=bh.storage.Int64())
    h[:] = (2, 4, 8)
    result = 1 / h
    assert result.storage_type is bh.storage.Double
    assert result.view() == approx(1 / np.array([2, 4, 8]))
    assert h.storage_type is bh.storage.Int64


def test_reflected_operators_weight_storage():
    # scalar - weighted-hist works (variance is preserved through the negation),
    # but scalar / weighted-hist is rejected: the reciprocal of a weighted sum
    # has no meaningful variance (matches test_view_rdiv_rejected).
    h = bh.Histogram(bh.axis.Regular(3, 0, 3), storage=bh.storage.Weight())
    h.fill([0, 1, 1, 2], weight=[1, 2, 3, 4])
    values = h.values().copy()

    assert (10 - h).values() == approx(10 - values)
    with pytest.raises(TypeError, match="divide a scalar or array"):
        12 / h


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

    with pytest.raises(ValueError):
        h.project(9)

    with pytest.raises(ValueError):
        h.project(-1)

    h_noflow = h.project(0, flow=False)
    assert [h_noflow[i] for i in range(2)] == [2, 1]

    h.fill(-1, 0)
    h.fill(0, 0)

    h_flow_true = h.project(0, flow=True)
    assert h_flow_true[0] == 3

    h_flow_false = h.project(0, flow=False)
    assert h_flow_false[0] == 2


def test_project_nd():
    h = bh.Histogram(
        bh.axis.Integer(0, 2), bh.axis.Integer(0, 2), bh.axis.Integer(0, 2)
    )

    h.fill(0, 0, 0)
    h.fill(1, 1, 0)

    h.fill(0, 0, -1)
    h.fill(1, 1, 2)

    h_flow_true = h.project(0, 1, flow=True)
    h_flow_false = h.project(0, 1, flow=False)

    assert h_flow_true[0, 0] == 2
    assert h_flow_true[1, 1] == 2

    assert h_flow_false[0, 0] == 1
    assert h_flow_false[1, 1] == 1


def test_shrink_1d():
    h = bh.Histogram(bh.axis.Regular(20, 1, 5))
    h.fill(1.1)
    hs = h[{0: slice(bh.loc(1), bh.loc(2))}]
    assert hs.view() == approx([1, 0, 0, 0, 0])

    d = OrderedDict({0: slice(bh.loc(1), bh.loc(2))})
    hs = h[d]
    assert hs.view() == approx([1, 0, 0, 0, 0])


@pytest.mark.parametrize(
    "metadata", [None, {}, {"a": "1"}], ids=["None", "empty", "dict"]
)
def test_rebin_1d(metadata):
    h = bh.Histogram(bh.axis.Regular(20, 1, 5, metadata=metadata))
    h.fill([1.1, 2.2, 3.3, 4.4])

    hs = h[{0: slice(None, None, bh.rebin(4))}]
    assert hs.view() == approx([1, 1, 1, 0, 1])
    assert h.axes[0].metadata is hs.axes[0].metadata

    hs = h[{0: bh.rebin(4)}]
    assert hs.view() == approx([1, 1, 1, 0, 1])
    assert h.axes[0].metadata is hs.axes[0].metadata

    hs = h[{0: bh.rebin(groups=[1, 2, 3, 14])}]
    assert hs.view() == approx([1, 0, 0, 3])
    assert hs.axes.edges[0] == approx([1.0, 1.2, 1.6, 2.2, 5.0])
    assert h.axes[0].metadata is hs.axes[0].metadata

    exact_edges = [1.0, 1.2, 1.6, 2.2, 5.0]
    hs = h[bh.rebin(edges=exact_edges)]
    assert hs.view() == approx([1, 0, 0, 3])
    assert hs.axes.edges[0] == approx(exact_edges)
    assert h.axes[0].metadata is hs.axes[0].metadata

    fuzzy_edges = [1.0, 1.200000000000001, 1.6, 2.2, 5.0]
    hs = h[bh.rebin(edges=fuzzy_edges)]
    assert hs.view() == approx([1, 0, 0, 3])
    assert hs.axes.edges[0] == approx(exact_edges)
    assert h.axes[0].metadata is hs.axes[0].metadata

    hs = h[bh.rebin(axis=bh.axis.Variable([1.0, 1.2, 1.6, 2.2, 5.0], metadata="hi"))]
    assert hs.view() == approx([1, 0, 0, 3])
    assert hs.axes.edges[0] == approx([1.0, 1.2, 1.6, 2.2, 5.0])
    assert hs.axes[0].metadata == "hi"


def test_rebin_1d_flow():
    h = bh.Histogram(bh.axis.Regular(5, 0, 5, underflow=True, overflow=True))
    h.fill([-1, 1.1, 2.2, 3.3, 4.4, 5.5])
    hs = h[bh.rebin(edges=[0, 3, 5.0])]
    assert hs.view() == approx([2, 2])
    assert hs.view(flow=True) == approx([1, 2, 2, 1])
    assert hs.axes.edges[0] == approx([0.0, 3.0, 5.0])

    # Flow bins are kept from the original
    h = bh.Histogram(bh.axis.Regular(5, 0, 5, underflow=False, overflow=False))
    h.fill([-1, 1.1, 2.2, 3.3, 4.4, 5.5])
    hs = h[bh.rebin(edges=[0, 3, 5.0])]
    assert hs.view(flow=True) == approx([2, 2])

    h = bh.Histogram(bh.axis.Regular(5, 0, 5, underflow=True, overflow=False))
    h.fill([-1, 1.1, 2.2, 3.3, 4.4, 5.5])
    hs = h[bh.rebin(edges=[0, 3, 5.0])]
    assert hs.view(flow=True) == approx([1, 2, 2])

    h = bh.Histogram(bh.axis.Regular(5, 0, 5, underflow=True, overflow=True))
    h.fill([-1, 1.1, 2.2, 3.3, 4.4, 5.5])
    hs = h[
        bh.rebin(axis=bh.axis.Variable([0, 3, 5.0], underflow=False, overflow=False))
    ]
    assert hs.view(flow=True) == approx([2, 2])


def test_rebin_change_axis_int():
    h = bh.Histogram(bh.axis.Regular(5, 0, 5))
    h.fill([-1, 1.1, 2.2, 3.3, 4.4, 5.5])
    hs = h[bh.rebin(edges=[0, 3, 5.0], axis=bh.axis.Integer(10, 12))]
    assert hs.view() == approx([2, 2])
    assert hs.view(flow=True) == approx([1, 2, 2, 1])
    assert hs.axes.edges[0] == approx([10, 11, 12])


def test_rebin_change_axis_cat():
    h = bh.Histogram(bh.axis.Regular(5, 0, 5))
    h.fill([-1, 1.1, 2.2, 3.3, 4.4, 5.5])
    hs = h[bh.rebin(groups=[2, 3], axis=bh.axis.StrCategory(["a", "b"]))]
    assert hs.view() == approx([1, 3])
    # Old underflow (1) and old overflow (1) both end up in the new overflow,
    # exactly once each.
    assert hs.view(flow=True) == approx([1, 3, 2])
    assert list(hs.axes[0]) == approx(["a", "b"])


def test_rebin_groups_flow_to_noflow():
    # Issue #1143 (B6b): rebinning a flow axis onto a categorical axis
    # without an underflow bin must not fold the old underflow into the
    # first group, and must add it to the new overflow exactly once.
    h = bh.Histogram(bh.axis.Regular(4, 0, 4))
    h.view(flow=True)[:] = [100, 1, 2, 3, 4, 200]
    hs = h[bh.rebin(groups=[2, 2], axis=bh.axis.IntCategory([0, 1]))]
    assert hs.view(flow=True) == approx([3, 7, 300])


def test_rebin_groups_flow_combinations():
    # All combinations of (old flow) x (new flow) for group rebinning.
    h = bh.Histogram(bh.axis.Regular(4, 0, 4))
    h.view(flow=True)[:] = [100, 1, 2, 3, 4, 200]

    # Old flow -> new flow: flow bins are carried over unchanged
    assert h[bh.rebin(groups=[2, 2])].view(flow=True) == approx([100, 3, 7, 200])

    # Old underflow dropped on an ordered axis without underflow
    hs = h[bh.rebin(groups=[2, 2], axis=bh.axis.Variable([0, 2, 4], underflow=False))]
    assert hs.view(flow=True) == approx([3, 7, 200])

    # Old overflow dropped on an axis without overflow
    hs = h[bh.rebin(groups=[2, 2], axis=bh.axis.Variable([0, 2, 4], overflow=False))]
    assert hs.view(flow=True) == approx([100, 3, 7])

    # Both flow bins dropped on a flowless ordered axis
    hs = h[
        bh.rebin(
            groups=[2, 2],
            axis=bh.axis.Variable([0, 2, 4], underflow=False, overflow=False),
        )
    ]
    assert hs.view(flow=True) == approx([3, 7])

    # Old axis without underflow: new axis copies the traits
    h2 = bh.Histogram(bh.axis.Regular(4, 0, 4, underflow=False))
    h2.view(flow=True)[:] = [1, 2, 3, 4, 200]
    assert h2[bh.rebin(groups=[2, 2])].view(flow=True) == approx([3, 7, 200])

    # Old axis without underflow onto a categorical axis: only the old
    # overflow goes to the new overflow
    hs = h2[bh.rebin(groups=[2, 2], axis=bh.axis.IntCategory([0, 1]))]
    assert hs.view(flow=True) == approx([3, 7, 200])

    # Old flowless axis onto an axis with flow: new flow bins stay empty
    h3 = bh.Histogram(bh.axis.Regular(4, 0, 4, underflow=False, overflow=False))
    h3.view(flow=True)[:] = [1, 2, 3, 4]
    hs = h3[bh.rebin(groups=[2, 2], axis=bh.axis.Variable([0, 2, 4]))]
    assert hs.view(flow=True) == approx([0, 3, 7, 0])


def test_shrink_rebin_1d():
    h = bh.Histogram(bh.axis.Regular(20, 0, 4))
    h.fill(1.1)
    hs = h[{0: slice(bh.loc(1), bh.loc(3), bh.rebin(2))}]
    assert hs.view() == approx([1, 0, 0, 0, 0])


def test_rebin_nd():
    h = bh.Histogram(
        bh.axis.Regular(20, 1, 3), bh.axis.Regular(30, 1, 3), bh.axis.Regular(40, 1, 3)
    )

    assert h[{0: np.s_[:: bh.rebin(2)]}].axes.size == (10, 30, 40)
    assert h[{1: np.s_[:: bh.rebin(2)]}].axes.size == (20, 15, 40)
    assert h[{2: np.s_[:: bh.rebin(2)]}].axes.size == (20, 30, 20)

    assert h[{0: np.s_[:: bh.rebin(groups=[1, 2, 17])]}].axes.size == (3, 30, 40)
    assert h[{1: np.s_[:: bh.rebin(groups=[1, 2, 27])]}].axes.size == (20, 3, 40)
    assert h[{2: np.s_[:: bh.rebin(groups=[1, 2, 37])]}].axes.size == (20, 30, 3)
    assert np.all(
        np.isclose(
            h[{0: np.s_[:: bh.rebin(groups=[1, 2, 17])]}].axes[0].edges,
            [1.0, 1.1, 1.3, 3.0],
        )
    )
    assert np.all(
        np.isclose(
            h[{1: np.s_[:: bh.rebin(groups=[1, 2, 27])]}].axes[1].edges,
            [1.0, 1.06666667, 1.2, 3.0],
        )
    )
    assert np.all(
        np.isclose(
            h[{2: np.s_[:: bh.rebin(groups=[1, 2, 37])]}].axes[2].edges,
            [1.0, 1.05, 1.15, 3.0],
        )
    )

    assert h[{0: np.s_[:: bh.rebin(2)], 2: np.s_[:: bh.rebin(2)]}].axes.size == (
        10,
        30,
        20,
    )

    assert h[
        {
            0: np.s_[:: bh.rebin(groups=[1, 2, 17])],
            2: np.s_[:: bh.rebin(groups=[1, 2, 37])],
        }
    ].axes.size == (3, 30, 3)
    assert np.all(
        np.isclose(
            h[
                {
                    0: np.s_[:: bh.rebin(groups=[1, 2, 17])],
                    2: np.s_[:: bh.rebin(groups=[1, 2, 37])],
                }
            ]
            .axes[0]
            .edges,
            [1.0, 1.1, 1.3, 3],
        )
    )
    assert np.all(
        np.isclose(
            h[
                {
                    0: np.s_[:: bh.rebin(groups=[1, 2, 17])],
                    2: np.s_[:: bh.rebin(groups=[1, 2, 37])],
                }
            ]
            .axes[2]
            .edges,
            [1.0, 1.05, 1.15, 3.0],
        )
    )

    assert h[{1: np.s_[:: bh.sum]}].axes.size == (20, 40)
    assert h[{1: bh.sum}].axes.size == (20, 40)


# CLASSIC: This used to have metadata too, but that does not compare equal
def test_pickle_0(flow):
    a = bh.Histogram(
        bh.axis.IntCategory([0, 1, 2], overflow=flow),
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
                for l in range(a.axes[3].extent):
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
                for l in range(a.axes[3].extent):
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

    assert h.view() == approx([1, 4])


def test_pick_bool():
    h = bh.Histogram(bh.axis.Boolean(), bh.axis.Boolean(metadata={"one": 1}))

    h.fill([True, True, False, False], [True, False, True, True])
    h.fill([True, True, True], True)

    assert h[True, :].view() == approx([1, 4])
    assert h[False, :].view() == approx([0, 2])
    assert h[:, False].view() == approx([0, 1])
    assert h[:, True].view() == approx([2, 4])


def test_slice_bool():
    h = bh.Histogram(bh.axis.Boolean())
    h.fill([0, 0, 0, 1, 3, 4, -2])

    assert h.view() == approx([3, 4])
    assert h[1:].view() == approx([4])
    assert h[:1].view() == approx([3])

    assert h[:1].axes[0].centers == approx([0.5])
    assert h[1:].axes[0].centers == approx([1.5])


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
    assert a.view() == approx(b.view())


# NumPy tests


def test_numpy_conversion_0():
    a = bh.Histogram(bh.axis.Integer(0, 3, underflow=False, overflow=False))
    a.fill(0)
    for _ in range(5):
        a.fill(1)
    c = np.array(a)  # a copy
    v = np.asarray(a)  # a view

    for t in (c, v):
        assert t.dtype == np.double  # CLASSIC: np.uint8
        assert t == approx((1, 5, 0))

    for _ in range(10):
        a.fill(2)
    # copy does not change, but view does
    assert c == approx((1, 5, 0))
    assert v == approx((1, 5, 10))

    for _ in range(255):
        a.fill(1)
    c = np.array(a)

    assert c.dtype == np.double  # CLASSIC: np.uint16
    assert c == approx((1, 260, 10))
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
    assert c == approx(np.array((0, 30, 0)))
    assert v == approx(c)


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

    assert d == approx(r)

    c = np.array(a)  # a copy
    v = np.asarray(a)  # a view

    assert c == approx(r)
    assert v == approx(r)


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

    assert c == approx(r)

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
    assert a1[0, 0] == float(2**80)
    assert a1[1, 0] == 1
    assert a1[2, 0] == 2
    assert a1[0, 1] == 3
    assert a1[1, 1] == 4
    assert a1[2, 1] == 5


def test_rank0_sum_empty():
    # A rank-0 histogram has a single cell and no flow bins. inner coverage
    # used to drive Boost's indexed range, which is UB for rank-0 and crashed
    # depending on stack layout (e.g. free-threaded Windows). inner == all here.
    h = bh.Histogram()
    assert h.ndim == 0
    assert h.empty() is True
    assert h.empty(flow=True) is True
    assert h.sum() == 0
    assert h.sum(flow=True) == 0

    h.fill()
    h.fill()
    h.fill()
    assert h.empty() is False
    assert h.sum() == 3
    assert h.sum(flow=True) == 3


def test_rank0_project():
    # project() drives the same rank-0-UB indexed range as sum()/empty() above,
    # and unlike those it cannot be avoided by picking a coverage. The identity
    # is the only valid projection of a rank-0 histogram.
    h = bh.Histogram()
    h.fill()
    assert h.project().ndim == 0
    assert h.project().sum() == 1


def test_rank0_reduce():
    # reduce() drives the same rank-0-UB indexed range. Nothing public reaches
    # it on a rank-0 histogram today (__getitem__ only reduces when it has
    # slices to apply), so exercise the binding directly.
    h = bh.Histogram()
    h.fill()
    assert h._reduce().ndim == 0
    assert h._reduce().sum() == 1


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64],
)
def test_fill_dtypes(dtype):
    a = bh.Histogram(bh.axis.Integer(0, 2), storage=bh.storage.Int64())
    a.fill(np.array([0, 0, 0, 1, 1, 2], dtype=dtype))
    a.fill(dtype(0))
    assert list(a.values()) == [4, 2]


def test_fill_with_sequence_0():
    def fa(*args):
        return np.array(args, dtype=float)

    def ia(*args):
        return np.array(args, dtype=int)

    a = bh.Histogram(bh.axis.Integer(0, 2))
    a.fill(np.array(1))  # 0-dim arrays work
    a.fill(ia(-1, 0, 1, 2))
    a.fill((2, 1, 0, -1))
    assert a.view(True) == approx([2, 2, 3, 2])

    with pytest.raises(ValueError):
        a.fill(np.empty((2, 2)))
    with pytest.raises(ValueError):
        a.fill(np.empty(2), 1)
    with pytest.raises(ValueError):
        a.fill(np.empty(2), np.empty(3))
    with pytest.raises(TypeError):
        a.fill("abc")

    with pytest.raises(IndexError):
        a[1, 2]

    b = bh.Histogram(bh.axis.Regular(3, 0, 3))
    b.fill(fa(0, 0, 1, 2))
    b.fill(ia(1, 0, 2, 2))
    assert b.view(True) == approx([0, 3, 2, 3, 0])

    c = bh.Histogram(
        bh.axis.Integer(0, 2, underflow=False, overflow=False), bh.axis.Regular(2, 0, 2)
    )
    c.fill(ia(-1, 0, 1), fa(-1.0, 1.5, 0.5))
    assert c.view(True) == approx(np.array([[0, 0, 1, 0], [0, 1, 0, 0]]))
    # we don't support: assert a[[1, 1]].value, 0

    with pytest.raises(ValueError):
        c.fill(1)
    with pytest.raises(ValueError):
        c.fill([1, 0, 2], [1, 1])

    # this broadcasts
    c.fill([1, 0], -1)
    assert c.view(True) == approx(np.array([[1, 0, 1, 0], [1, 1, 0, 0]]))
    c.fill([1, 0], 0)
    assert c.view(True) == approx(np.array([[1, 1, 1, 0], [1, 2, 0, 0]]))
    c.fill(0, [-1, 0.5, 1.5, 2.5])
    assert c.view(True) == approx(np.array([[2, 2, 2, 1], [1, 2, 0, 0]]))

    with pytest.raises(IndexError):
        c[1]
    with pytest.raises(IndexError):
        c[1, 2, 3]


@pytest.mark.xfail(
    platform.machine() == "ppc64le", reason="ppc64le bug (TBD)", strict=False
)
def test_fill_with_sequence_1():
    a = bh.Histogram(bh.axis.Integer(0, 3), storage=bh.storage.Weight())
    v = np.array([-1, 0, 1, 2, 3, 4], dtype=int)
    w = np.array([2, 3, 4, 5, 6, 7], dtype=float)
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
    assert a.view(True) == approx([2, 2, 1])
    a.fill(np.array(("D", "B", "A"), dtype="S5"))
    a.fill(np.array(("D", "B", "A"), dtype="U1"))
    assert a.view(True) == approx([4, 4, 3])

    with pytest.raises(ValueError):
        a.fill(np.array((("B", "A"), ("C", "A"))))  # ndim == 2 not allowed

    b = bh.Histogram(
        bh.axis.Integer(0, 2, underflow=False, overflow=False),
        bh.axis.StrCategory(["A", "B"]),
    )
    b.fill((1, 0, 10), ("C", "B", "A"))
    assert b.view(True) == approx(np.array([[0, 1, 0], [0, 0, 1]]))


def test_fill_with_sequence_3():
    h = bh.Histogram(bh.axis.StrCategory([], growth=True))
    h.fill("A")
    assert h.axes[0].size == 1
    h.fill(["A"])
    assert h.axes[0].size == 1
    h.fill(["A", "B"])
    assert h.axes[0].size == 2
    assert h.view(True) == approx([3, 1])


def test_fill_with_sequence_4():
    h = bh.Histogram(
        bh.axis.StrCategory([], growth=True), bh.axis.Integer(0, 0, growth=True)
    )
    h.fill("1", np.arange(2))
    assert h.axes[0].size == 1
    assert h.axes[1].size == 2
    assert h.view(True) == approx(np.array([[1, 1]]))

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

    if platform.python_implementation() == "CPython":
        # 2 is the minimum refcount, so the *python* object should be deleted
        # after the del; hopefully the C++ object lives through the axis instance.
        # On Python 3.14, this can be 1
        assert sys.getrefcount(h) <= 2

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

    assert h1.view() == approx(dens)


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

    assert np.asarray(h) == approx(1)
    assert np.asarray(h1) == approx(2)
    assert np.asarray(h2) == approx(3)
    assert np.asarray(h3) == approx(6)


def test_add_broadcast():
    h = bh.Histogram(bh.axis.Regular(10, 0, 1), bh.axis.Regular(20, 0, 1))

    h1 = h.copy()
    h2 = h.copy()

    h1[...] = 1
    assert h1.view().sum() == 10 * 20
    assert h1.view(flow=True).sum() == 10 * 20

    h2 = h + [[1]]  # noqa: RUF005
    assert h2.sum() == 10 * 20
    assert h2.sum(flow=True) == 12 * 22

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

    assert widths_1 == approx(widths_2)


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


def test_sum_empty_axis():
    hist = bh.Histogram(
        bh.axis.StrCategory("", growth=True),
        bh.axis.Regular(10, 0, 1),
        storage=bh.storage.Weight(),
    )
    assert hist.sum().value == 0
    assert "Str" in repr(hist)


# Issue #618
def test_negative_fill(count_storage):
    h = bh.Histogram(bh.axis.Integer(0, 3), storage=count_storage())
    h.fill(1, weight=-1)

    answer = np.array([0, -1, 0])
    assert h.values() == approx(answer)


# Issue #589
def test_underfill_growth():
    h = bh.Histogram(bh.axis.Regular(10, 0, 1, growth=True))
    h.fill(2)
    h.fill(-1)
    assert h.sum() == 2


# ---- allclose tests ---------------------------------------------------------


def test_allclose_same_histogram():
    h = bh.Histogram(bh.axis.Regular(5, 0, 1))
    h.fill(np.random.default_rng(42).random(100))

    assert h.allclose(h)
    assert h.allclose(h.copy())


def test_allclose_different_bins():
    h1 = bh.Histogram(bh.axis.Regular(5, 0, 1))
    h2 = bh.Histogram(bh.axis.Regular(5, 0, 1))

    rng = np.random.default_rng(42)
    vals = rng.random(100)
    h1.fill(vals)
    h2.fill(vals)
    h2.fill(0.7)

    assert not h1.allclose(h2)


def test_allclose_edges_within_tol():
    h1 = bh.Histogram(bh.axis.Regular(5, 0, 1))
    h2 = bh.Histogram(bh.axis.Regular(5, 0, 1 + 1e-7))

    assert h1.allclose(h2)


def test_allclose_edges_outside_tol():
    h1 = bh.Histogram(bh.axis.Regular(5, 0, 1))
    h2 = bh.Histogram(bh.axis.Regular(5, 0, 1 + 0.1))

    assert not h1.allclose(h2)


def test_allclose_different_shape():
    h1 = bh.Histogram(bh.axis.Regular(5, 0, 1))
    h2 = bh.Histogram(bh.axis.Regular(6, 0, 1))

    assert not h1.allclose(h2)


def test_allclose_different_ndim():
    h1 = bh.Histogram(bh.axis.Regular(5, 0, 1))
    h2 = bh.Histogram(bh.axis.Regular(5, 0, 1), bh.axis.Regular(3, 0, 1))

    assert not h1.allclose(h2)


def test_allclose_different_storage():
    h1 = bh.Histogram(bh.axis.Regular(5, 0, 1))
    h2 = bh.Histogram(bh.axis.Regular(5, 0, 1), storage=bh.storage.Weight())

    assert not h1.allclose(h2)


def test_allclose_flow_option():
    h1 = bh.Histogram(bh.axis.Regular(3, 0, 1))
    h2 = bh.Histogram(bh.axis.Regular(3, 0, 1))

    h1.fill([0.5])  # central bin only
    h2.fill([0.5])  # central bin only
    # h1 has flow bins empty, h2 also, so both should match with/without flow
    assert h1.allclose(h2, flow=True)
    assert h1.allclose(h2, flow=False)

    # Now put different data in underflow
    h3 = bh.Histogram(bh.axis.Regular(3, 0, 1))
    h4 = bh.Histogram(bh.axis.Regular(3, 0, 1))
    h3.fill([-0.5])
    h4.fill([-0.5, -0.5])

    assert not h3.allclose(h4, flow=True)
    assert h3.allclose(h4, flow=False)


def test_allclose_metadata():
    h1 = bh.Histogram(bh.axis.Regular(3, 0, 1), metadata="foo")
    h2 = bh.Histogram(bh.axis.Regular(3, 0, 1), metadata="foo")
    h3 = bh.Histogram(bh.axis.Regular(3, 0, 1), metadata="bar")

    assert h1.allclose(h2)
    assert h1.allclose(h2, metadata=True)
    assert not h1.allclose(h3, metadata=True)
    assert h1.allclose(h3, metadata=False)


def test_allclose_non_histogram():
    h = bh.Histogram(bh.axis.Regular(3, 0, 1))
    assert not h.allclose(42)
    assert not h.allclose("histogram")


def test_allclose_categorical_exact():
    h1 = bh.Histogram(bh.axis.StrCategory(["a", "b"]))
    h2 = bh.Histogram(bh.axis.StrCategory(["a", "b"]))
    h3 = bh.Histogram(bh.axis.StrCategory(["a", "c"]))
    h4 = bh.Histogram(bh.axis.StrCategory(["b", "a"]))

    assert h1.allclose(h2)
    assert not h1.allclose(h3)
    assert not h1.allclose(h4)


def test_allclose_intcategory():
    h1 = bh.Histogram(bh.axis.IntCategory([1, 2, 3]))
    h2 = bh.Histogram(bh.axis.IntCategory([1, 2, 3]))
    h3 = bh.Histogram(bh.axis.IntCategory([1, 2, 4]))

    assert h1.allclose(h2)
    assert not h1.allclose(h3)


def test_allclose_boolean():
    h1 = bh.Histogram(bh.axis.Boolean())
    h2 = bh.Histogram(bh.axis.Boolean())

    assert h1.allclose(h2)


def test_allclose_mean_storage():
    h1 = bh.Histogram(bh.axis.Regular(3, 0, 1), storage=bh.storage.Mean())
    h2 = bh.Histogram(bh.axis.Regular(3, 0, 1), storage=bh.storage.Mean())

    rng = np.random.default_rng(42)
    samples = rng.random(100)
    h1.fill(samples, sample=samples)
    h2.fill(samples, sample=samples)

    assert h1.allclose(h2)


def test_allclose_weight_storage():
    h1 = bh.Histogram(bh.axis.Regular(3, 0, 1), storage=bh.storage.Weight())
    h2 = bh.Histogram(bh.axis.Regular(3, 0, 1), storage=bh.storage.Weight())

    rng = np.random.default_rng(42)
    data = rng.random(100)
    w = rng.random(100)
    h1.fill(data, weight=w)
    h2.fill(data, weight=w)

    assert h1.allclose(h2)


def test_allclose_weighted_mean_storage():
    h1 = bh.Histogram(bh.axis.Regular(3, 0, 1), storage=bh.storage.WeightedMean())
    h2 = bh.Histogram(bh.axis.Regular(3, 0, 1), storage=bh.storage.WeightedMean())

    rng = np.random.default_rng(42)
    data = rng.random(100)
    w = rng.random(100)
    h1.fill(data, sample=data, weight=w)
    h2.fill(data, sample=data, weight=w)

    assert h1.allclose(h2)


def test_allclose_different_continuous_traits():
    h1 = bh.Histogram(bh.axis.Regular(3, 0, 1))
    h2 = bh.Histogram(bh.axis.Integer(0, 3))

    assert not h1.allclose(h2)


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="needs subprocess")
def test_missing_core_hint():
    # Issue #1143 (B10): if the compiled _core module is missing, the import
    # error must include the "did you forget to compile" hint. This only works
    # if boost_histogram/__init__.py imports _core before any submodule does.
    code = """
import sys
import importlib.abc

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "boost_histogram._core":
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None

sys.meta_path.insert(0, Blocker())
import boost_histogram
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "Did you forget to compile boost-histogram?" in result.stderr


def test_fill_noncontiguous_int_category():
    # Regression test for gh-1143: fill must honor input strides
    cat = [1, 2, 3]
    expected = bh.Histogram(bh.axis.IntCategory(cat))
    expected.fill([1, 3, 2, 1])

    h = bh.Histogram(bh.axis.IntCategory(cat))
    h.fill(np.array([1, 2, 3, 1])[::-1])
    assert h.view(flow=True) == approx(expected.view(flow=True))

    h2 = bh.Histogram(bh.axis.IntCategory(cat))
    h2.fill(np.array([1, 7, 3, 7, 2, 7, 1, 7])[::2])
    assert h2.view(flow=True) == approx(expected.view(flow=True))


def test_fill_noncontiguous_str_category():
    # Regression test for gh-1143: fill must honor input strides
    cat = ["a", "b", "c"]
    expected = bh.Histogram(bh.axis.StrCategory(cat))
    expected.fill(["a", "c", "a"])

    h = bh.Histogram(bh.axis.StrCategory(cat))
    h.fill(np.array(["a", "b", "c", "b", "a"])[::2])
    assert h.view(flow=True) == approx(expected.view(flow=True))

    expected2 = bh.Histogram(bh.axis.StrCategory(cat))
    expected2.fill(["c", "b", "a"])

    h2 = bh.Histogram(bh.axis.StrCategory(cat))
    h2.fill(np.array(["a", "b", "c"])[::-1])
    assert h2.view(flow=True) == approx(expected2.view(flow=True))


@pytest.mark.parametrize(
    ("dtype", "value"),
    [(np.int64, 2**32 + 3), (np.uint32, 2**31 + 3), (np.uint64, 2**32 + 3)],
)
def test_fill_int_axis_out_of_range(dtype, value):
    # Regression test: wide integers were narrowed to int32 and wrapped silently
    h = bh.Histogram(bh.axis.Integer(0, 5))
    with pytest.raises(ValueError):
        h.fill(np.array([value], dtype=dtype))
    assert h.sum(flow=True) == 0

    c = bh.Histogram(bh.axis.IntCategory([1, 2, 3]))
    with pytest.raises(ValueError):
        c.fill(np.array([value], dtype=dtype))
    assert c.sum(flow=True) == 0


def test_index_int_axis_out_of_range():
    with pytest.raises(ValueError):
        bh.axis.Integer(0, 5).index(np.array([2**32 + 3]))
    with pytest.raises(ValueError):
        bh.axis.IntCategory([1, 2, 3]).index(np.array([2**32 + 2]))


def test_fill_int_axis_scalar_out_of_range():
    h = bh.Histogram(bh.axis.Integer(0, 5))
    with pytest.raises((ValueError, TypeError)):
        h.fill(2**32 + 3)
    assert h.sum(flow=True) == 0


@pytest.mark.parametrize(
    "dtype", [np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.int64]
)
def test_fill_int_axis_dtypes(dtype):
    h = bh.Histogram(bh.axis.Integer(0, 5))
    h.fill(np.array([0, 1, 1], dtype=dtype))
    assert h.view() == approx(np.array([1, 2, 0, 0, 0]))

    b = bh.Histogram(bh.axis.Boolean())
    b.fill(np.array([0, 1, 1], dtype=dtype))
    assert b.view() == approx(np.array([1, 2]))


def test_compare_ndarray_metadata():
    # Metadata that gives no plain bool from == must not abort the interpreter
    array = np.arange(3)
    h = bh.Histogram(bh.axis.Regular(3, 0, 1, metadata=array))
    h_same = bh.Histogram(bh.axis.Regular(3, 0, 1, metadata=array))
    h_other = bh.Histogram(bh.axis.Regular(3, 0, 1, metadata=np.arange(3)))

    assert h == h_same
    assert h != h_other
    assert (h + h_same).sum() == 0
    assert h.copy(deep=False) == h

    with pytest.raises(ValueError, match="axes"):
        h + h_other


@pytest.mark.parametrize(
    "axis_type", [bh.axis.StrCategory, bh.axis.IntCategory], ids=["str", "int"]
)
def test_growth_axis_survives_merge(axis_type):
    # An axis held by the user must not dangle when a merge replaces it
    values = ["a", "b", "c"] if axis_type is bh.axis.StrCategory else [1, 2, 3]

    h = bh.Histogram(axis_type([], growth=True))
    h.fill(values[:1])
    ax = h.axes[0]
    assert ax.size == 1

    h2 = bh.Histogram(axis_type([], growth=True))
    h2.fill(values[1:])
    h += h2

    # The old axis is a snapshot of the merged-from state
    assert ax.size == 1
    assert ax.index(values[0]) == 0

    assert h.axes[0].size == 3
    assert list(h.axes[0]) == values


def test_growth_axis_size_after_fill():
    h = bh.Histogram(bh.axis.Regular(2, 0, 1, growth=True))
    assert h.axes[0].size == 2
    h.fill([0.5, 2.5])
    assert h.axes[0].size == 6
    assert h.axes[0].size == h.view().size


def test_view_after_growth():
    # A growing fill moves the storage; the view must follow the new size
    h = bh.Histogram(bh.axis.Integer(0, 5, growth=True))
    h.fill(np.arange(1000))

    assert h.view().sum() == 1000
    assert h.view(flow=True).size == 1000


def test_setitem_with_growth_axis():
    h = bh.Histogram(bh.axis.Regular(4, 0, 1, growth=True))
    h[...] = np.arange(4)
    assert h.view() == approx(np.arange(4))

    h[1] = 10
    assert h[1] == 10
    assert h.view() == approx([0, 10, 2, 3])

    h[np.array([0, 2])] = [5, 6]
    assert h.view() == approx([5, 10, 6, 3])


def test_inplace_op_with_growth_axis():
    h = bh.Histogram(bh.axis.Regular(4, 0, 1, growth=True))
    h.fill([0.1, 0.6])
    h *= 2
    assert h.sum() == 4
    h /= 2
    assert h.sum() == 2
    assert h.axes[0].size == 4


@pytest.mark.parametrize("deep", [False, True], ids=["shallow", "deep"])
def test_copy_growth_axis_metadata_is_independent(deep):
    h = bh.Histogram(bh.axis.Regular(4, 0, 1, growth=True, metadata="original"))
    h2 = h.copy(deep=deep)
    h2.axes[0].metadata = "changed"

    assert h.axes[0].metadata == "original"
    assert h2.axes[0].metadata == "changed"


def test_growth_axis_metadata_write_through():
    h = bh.Histogram(bh.axis.StrCategory([], growth=True))
    h.axes[0].metadata = "label"
    assert h.axes[0].metadata == "label"
    h.fill(["a"])
    assert h.axes[0].metadata == "label"

    # The copy of a growth axis must hold the same dict as the stored axis
    assert h.axes[0].__dict__ is h.axes[0].__dict__
