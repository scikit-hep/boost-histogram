# -*- coding: utf-8 -*-
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import boost_histogram as bh


def test_1D_get_bin():

    h = bh.Histogram(bh.axis.Regular(10, 0, 0.99))
    h.fill([0.25, 0.25, 0.25, 0.15])

    assert h[0] == 0
    assert h[1] == 1
    assert h[2] == 3

    assert h[bh.loc(0)] == 0
    assert h[bh.loc(0.1)] == 1
    assert h[bh.loc(0.1) + 1] == 3
    assert h[bh.loc(0.2)] == 3

    assert h.view()[2] == h[2]

    with pytest.raises(IndexError):
        h[1, 2]


def test_2D_get_bin():

    h = bh.Histogram(bh.axis.Regular(10, 0, 0.99), bh.axis.Regular(10, 0, 0.99))
    h.fill(0.15, [0.25, 0.25, 0.25, 0.15])

    assert h[0, 0] == 0
    assert h[0, 1] == 0
    assert h[1, 1] == 1
    assert h[1, 2] == 3
    assert h[bh.loc(0.1), bh.loc(0.2)] == 3
    assert h[bh.loc(0) + 1, bh.loc(0.3) - 1] == 3

    assert h.view()[1, 2] == h[1, 2]

    with pytest.raises(IndexError):
        h[1]


def test_get_1D_histogram():
    h = bh.Histogram(bh.axis.Regular(10, 0, 1))
    h.fill([0.25, 0.25, 0.25, 0.15])

    h2 = h[:]

    assert h == h2

    h.fill(0.75)

    assert h != h2


def test_get_1D_slice():
    h1 = bh.Histogram(bh.axis.Regular(10, 0, 1))
    h2 = bh.Histogram(bh.axis.Regular(5, 0, 0.5))
    h1.metadata = {"that": 3}

    h1.fill([0.25, 0.25, 0.25, 0.15])
    h2.fill([0.25, 0.25, 0.25, 0.15])

    assert h1 != h2
    assert h1[:5] == h2
    assert h1[: bh.loc(0.5)] == h2
    assert h1[2:4] == h2[2:4]
    assert h1[bh.loc(0.2) : bh.loc(0.4)] == h2[bh.loc(0.2) : bh.loc(0.4)]

    assert len(h1[2:4].view()) == 2
    assert len(h1[2 : 4 : bh.rebin(2)].view()) == 1
    assert len(h1[:: bh.rebin(2)].view()) == 5

    # Shortcut
    assert len(h1[bh.rebin(2)].view()) == 5

    assert h1[2:4].metadata == {"that": 3}


def test_ellipsis():

    h = bh.Histogram(bh.axis.Regular(10, 0, 1), bh.axis.Regular(10, 0, 1))

    assert h == h[...]
    assert h == h[:, ...]
    assert h == h[..., :]
    assert h == h[:, :, ...]
    assert h == h[:, ..., :]
    assert h == h[..., :, :]

    with pytest.raises(IndexError):
        h[:, :, :, ...]
    with pytest.raises(IndexError):
        h[:, :, ..., :]
    with pytest.raises(IndexError):
        h[..., :, :, :]
    with pytest.raises(IndexError):
        h[..., ...]

    assert h[2:4, ...] == h[2:4, :]


def test_basic_projection():
    h2 = bh.Histogram(
        bh.axis.Regular(10, 0, 10),
        bh.axis.Regular(10, 0, 10),
        bh.axis.Regular(10, 0, 10),
    )
    h1 = bh.Histogram(bh.axis.Regular(10, 0, 10))

    contents = [[2, 2, 2, 3, 4, 5, 6], [1, 2, 2, 3, 2, 1, 2], [-12, 33, 4, 9, 2, 4, 9]]

    h1.fill(contents[0])
    h2.fill(*contents)

    assert h1 == h2[:, :: bh.sum, :: bh.sum]
    assert h1 == h2[..., :: bh.sum, :: bh.sum]
    assert h2.sum(flow=True) == h2[:: bh.sum, :: bh.sum, :: bh.sum]

    # Python's builtin sum is identical to bh.sum
    assert bh.sum is sum
    assert h1 == h2[:, ::sum, ::sum]
    assert h1 == h2[..., ::sum, ::sum]
    assert h2.sum(flow=True) == h2[::sum, ::sum, ::sum]

    # Shortcut
    assert h1 == h2[:, sum, sum]
    assert h1 == h2[..., sum, sum]
    assert h2.sum(flow=True) == h2[sum, sum, sum]


def test_slicing_projection():
    h1 = bh.Histogram(
        bh.axis.Regular(10, 0, 10),
        bh.axis.Regular(10, 0, 10),
        bh.axis.Regular(10, 0, 10),
    )

    X, Y, Z = np.mgrid[-0.5:10.5:12j, -0.5:10.5:12j, -0.5:10.5:12j]

    h1.fill(X.ravel(), Y.ravel(), Z.ravel())

    assert h1[:: bh.sum, :: bh.sum, :: bh.sum] == 12 ** 3
    assert h1[0 : len : bh.sum, 0 : len : bh.sum, 0 : len : bh.sum] == 10 ** 3
    assert h1[0 : bh.overflow : bh.sum, 0 : len : bh.sum, :: bh.sum] == 10 * 10 * 12
    assert h1[:: bh.sum, 0 : len : bh.sum, :: bh.sum] == 10 * 12 * 12

    # make sure nothing was modified
    assert h1.sum() == 10 ** 3
    assert h1.sum(flow=True) == 12 ** 3

    h2 = h1[0 : 3 : bh.sum, ...]
    assert h2[1, 2] == 3

    h3 = h2[:, 5 : 7 : bh.sum]
    assert h3[1] == 6

    # Select one bin
    assert h1[2, :: bh.sum, :: bh.sum] == 12 * 12

    # Select one bin
    assert h1[2, 7, :: bh.sum] == 12


def test_mix_value_with_slice():
    h = bh.Histogram(
        bh.axis.Regular(10, 0, 10), bh.axis.Regular(10, 0, 10), bh.axis.Integer(0, 2)
    )

    vals = np.arange(100).reshape(10, 10, 1)
    h[:, :, 1:2] = vals

    print(h.view()[:3, :3, :])

    assert h[0, 1, True] == 1
    assert h[1, 0, True] == 10
    assert h[1, 1, True] == 11
    assert h[3, 4, False] == 0

    assert_array_equal(h[:, :, True].view(), vals[:, :, 0])
    assert_array_equal(h[:, :, False].view(), 0)


def test_mix_value_with_slice_2():
    h = bh.Histogram(
        bh.axis.Regular(10, 0, 10), bh.axis.Regular(10, 0, 10), bh.axis.Integer(0, 2)
    )

    vals = np.arange(100).reshape(10, 10)
    h[:, :, True] = vals

    assert h[0, 1, True] == 1
    assert h[1, 0, True] == 10
    assert h[1, 1, True] == 11
    assert h[3, 4, False] == 0

    assert_array_equal(h[:, :, True].view(), vals)
    assert_array_equal(h[:, :, False].view(), 0)

    h2 = h[bh.rebin(2), bh.rebin(5), :]
    assert_array_equal(h2.shape, (5, 2, 2))


def test_one_sided_slice():
    h = bh.Histogram(bh.axis.Regular(4, 1, 5))
    h.view(True)[:] = 1

    assert h[sum] == 6  # 4 (internal bins) + 2 (flow bins)
    assert h[bh.tag.at(-1) : bh.tag.at(5) : sum] == 6  # keeps underflow, keeps overflow

    # check that slicing without bh.sum adds removed counts to flow bins
    assert_array_equal(h[1:3].view(True), [2, 1, 1, 2])

    assert h[0::sum] == 5  # removes underflow, keeps overflow
    assert h[:4:sum] == 5  # removes overflow, keeps underflow
    assert h[0:4:sum] == 4  # removes underflow and overflow

    assert h[bh.loc(1) :: sum] == 5  # remove underflow
    assert h[: bh.loc(5) : sum] == 5  # remove overflow
    assert h[bh.loc(1) : bh.loc(5) : sum] == 4  # removes underflow and overflow

    assert h[bh.loc(0) :: sum] == 6  # keep underflow
    assert h[: bh.loc(10) + 1 : sum] == 6  # keep overflow
    assert h[bh.loc(0) : bh.loc(10) + 1 : sum] == 6


def test_repr():
    assert repr(bh.loc(2)) == "loc(2)"
    assert repr(bh.loc(3) + 1) == "loc(3) + 1"
    assert repr(bh.loc(1) - 2) == "loc(1) - 2"

    assert repr(bh.underflow) == "underflow"
    assert repr(bh.underflow + 1) == "underflow + 1"
    assert repr(bh.underflow - 1) == "underflow - 1"

    assert repr(bh.overflow) == "overflow"
    assert repr(bh.overflow + 1) == "overflow + 1"
    assert repr(bh.overflow - 1) == "overflow - 1"

    assert repr(bh.rebin(2)) == "rebin(2)"


# Was broken in 0.6.1
def test_noflow_slicing():
    noflow = dict(underflow=False, overflow=False)

    h = bh.Histogram(
        bh.axis.Regular(10, 0, 10),
        bh.axis.Regular(10, 0, 10, **noflow),
        bh.axis.Integer(0, 2, **noflow),
    )

    vals = np.arange(100).reshape(10, 10)
    h[:, :, True] = vals

    assert h[0, 1, True] == 1
    assert h[1, 0, True] == 10
    assert h[1, 1, True] == 11
    assert h[3, 4, False] == 0
    assert h[{0: 3, 1: 4, 2: False}] == 0

    assert_array_equal(h[:, :, True].view(), vals)
    assert_array_equal(h[:, :, False].view(), 0)


def test_singleflow_slicing():
    h = bh.Histogram(
        bh.axis.Integer(0, 4, underflow=False), bh.axis.Integer(0, 4, overflow=False)
    )

    vals = np.arange(4 * 4).reshape(4, 4)
    h[:, :] = vals

    assert h[0, 0] == 0
    assert h[0, 1] == 1
    assert h[1, 0] == 4
    assert h[1, 1] == 5

    assert_array_equal(h[:, 1 : 3 : bh.sum], vals[:, 1:3].sum(axis=1))
    assert_array_equal(h[{1: slice(1, 3, bh.sum)}], vals[:, 1:3].sum(axis=1))
    assert_array_equal(h[1 : 3 : bh.sum, :], vals[1:3, :].sum(axis=0))


def test_pick_str_category():
    noflow = dict(underflow=False, overflow=False)

    h = bh.Histogram(
        bh.axis.Regular(10, 0, 10),
        bh.axis.Regular(10, 0, 10, **noflow),
        bh.axis.StrCategory(["on", "off", "maybe"]),
    )

    vals = np.arange(100).reshape(10, 10)
    h[:, :, bh.loc("on")] = vals

    assert h[0, 1, bh.loc("on")] == 1
    assert h[1, 0, bh.loc("on")] == 10
    assert h[1, 1, bh.loc("on")] == 11
    assert h[3, 4, bh.loc("maybe")] == 0

    assert_array_equal(h[:, :, bh.loc("on")].view(), vals)
    assert_array_equal(h[{2: bh.loc("on")}].view(), vals)
    assert_array_equal(h[:, :, bh.loc("off")].view(), 0)


def test_string_requirement():
    h = bh.Histogram(
        bh.axis.Integer(0, 10),
        bh.axis.StrCategory(["1", "a", "hello"]),
        storage=bh.storage.Int64(),
    )

    with pytest.raises(TypeError):
        h[bh.loc("1"), bh.loc(1)]

    with pytest.raises(TypeError):
        h[bh.loc(1), bh.loc(1)]

    with pytest.raises(TypeError):
        h[bh.loc("1"), bh.loc("1")]

    assert h[bh.loc(1), bh.loc("1")] == 0


def test_pick_int_category():
    noflow = dict(underflow=False, overflow=False)

    h = bh.Histogram(
        bh.axis.Regular(10, 0, 10),
        bh.axis.Regular(10, 0, 10, **noflow),
        bh.axis.IntCategory([3, 5, 7, 12, 13]),
    )

    vals = np.arange(100).reshape(10, 10)
    h[:, :, bh.loc(3)] = vals
    h[:, :, bh.loc(5)] = vals + 1
    h[:, :, 3] = vals + 100

    assert h[0, 1, bh.loc(3)] == 1
    assert h[1, 0, bh.loc(5)] == 10 + 1
    assert h[1, 1, bh.loc(5)] == 11 + 1
    assert h[3, 4, bh.loc(7)] == 0
    assert h[3, 4, bh.loc(12)] == 134

    assert_array_equal(h[:, :, bh.loc(3)].view(), vals)
    assert_array_equal(h[{2: bh.loc(3)}].view(), vals)
    assert_array_equal(h[:, :, bh.loc(5)].view(), vals + 1)
    assert_array_equal(h[:, :, bh.loc(7)].view(), 0)


def test_axes_tuple():
    h = bh.Histogram(bh.axis.Regular(10, 0, 1))
    assert isinstance(h.axes[:1], bh._internal.axestuple.AxesTuple)
    assert isinstance(h.axes[0], bh.axis.Regular)

    (before,) = h.axes.centers[:1]
    (after,) = h.axes[:1].centers

    assert_array_equal(before, after)


def test_axes_tuple_Nd():
    h = bh.Histogram(
        bh.axis.Integer(0, 5), bh.axis.Integer(0, 4), bh.axis.Integer(0, 6)
    )
    assert isinstance(h.axes[:2], bh._internal.axestuple.AxesTuple)
    assert isinstance(h.axes[1], bh.axis.Integer)

    b1, b2 = h.axes.centers[1:3]
    a1, a2 = h.axes[1:3].centers

    assert_array_equal(b1.flatten(), a1.flatten())
    assert_array_equal(b2.flatten(), a2.flatten())

    assert b1.ndim == 3
    assert a1.ndim == 2
