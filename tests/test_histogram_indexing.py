from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

import boost_histogram as bh
from boost_histogram import _core


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

    assert h1[:: bh.sum, :: bh.sum, :: bh.sum] == 12**3
    assert h1[0 : len : bh.sum, 0 : len : bh.sum, 0 : len : bh.sum] == 10**3
    assert h1[0 : bh.overflow : bh.sum, 0 : len : bh.sum, :: bh.sum] == 10 * 10 * 12
    assert h1[:: bh.sum, 0 : len : bh.sum, :: bh.sum] == 10 * 12 * 12

    # make sure nothing was modified
    assert h1.sum() == 10**3
    assert h1.sum(flow=True) == 12**3

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

    assert h[0, 1, True] == 1
    assert h[1, 0, True] == 10
    assert h[1, 1, True] == 11
    assert h[3, 4, False] == 0

    assert h[:, :, True].view() == approx(vals[:, :, 0])
    assert h[:, :, False].view() == approx(0)


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

    assert h[:, :, True].view() == approx(vals)
    assert h[:, :, False].view() == approx(0)

    h2 = h[bh.rebin(2), bh.rebin(5), :]
    assert h2.shape == approx((5, 2, 2))


def test_one_sided_slice():
    h = bh.Histogram(bh.axis.Regular(4, 1, 5))
    h.view(True)[:] = 1

    assert h[sum] == 6  # 4 (internal bins) + 2 (flow bins)
    assert h[bh.tag.at(-1) : bh.tag.at(5) : sum] == 6  # keeps underflow, keeps overflow

    # check that slicing without bh.sum adds removed counts to flow bins
    assert h[1:3].view(True) == approx([2, 1, 1, 2])

    assert h[0::sum] == 5  # removes underflow, keeps overflow
    assert h[:4:sum] == 5  # removes overflow, keeps underflow
    assert h[0:4:sum] == 4  # removes underflow and overflow

    assert h[bh.loc(1) :: sum] == 5  # remove underflow
    assert h[: bh.loc(5) : sum] == 5  # remove overflow
    assert h[bh.loc(1) : bh.loc(5) : sum] == 4  # removes underflow and overflow

    assert h[bh.loc(0) :: sum] == 6  # keep underflow
    assert h[: bh.loc(10) + 1 : sum] == 6  # keep overflow
    assert h[bh.loc(0) : bh.loc(10) + 1 : sum] == 6


def test_negative_index_access():
    # Negative plain-int indices and slice bounds are normalized relative to the
    # number of bins on the axis, like Python/NumPy semantics.
    h = bh.Histogram(bh.axis.Regular(10, 0, 10))
    h.fill(range(10))

    # Bare integer index already worked, but check it stays consistent
    assert h[-1] == h[9]
    assert h[-2] == h[8]

    # Slice bounds (positional)
    assert h[1:-2] == h[1:8]
    assert h[-3:-1] == h[7:9]
    assert h[-3:] == h[7:]
    assert h[:-2] == h[:8]

    # Dict indexing goes through the same path
    assert h[{0: -2}] == h[8]
    assert h[{0: slice(1, -2)}] == h[1:8]

    # Two-axis case to confirm per-axis normalization uses the right length
    h2 = bh.Histogram(bh.axis.Regular(10, 0, 10), bh.axis.Regular(4, 0, 4))
    assert h2[1:-2, :].axes[0].size == h2[1:8, :].axes[0].size
    assert h2[:, -1] == h2[:, 3]


def test_out_of_range_slice_clamps_like_numpy():
    # Out-of-range bounds clamp to [0, len] like NumPy, instead of raising or
    # accidentally pulling in flow bins.
    h = bh.Histogram(bh.axis.Regular(10, 0, 10))
    h.fill(range(10))

    a = np.arange(10)
    for sl in [
        slice(-15, None),
        slice(-15, 5),
        slice(5, 100),
        slice(None, 100),
        slice(-100, 8),
        slice(2, 100),
    ]:
        assert h[sl].axes[0].size == len(a[sl])


def test_empty_slice_raises():
    # NumPy returns an empty array; a Boost.Histogram axis cannot have zero
    # bins, so these raise a clear IndexError rather than a low-level error.
    h = bh.Histogram(bh.axis.Regular(10, 0, 10))
    h.fill(range(10))

    for sl in [
        slice(5, 2),
        slice(3, 3),
        slice(1, -15),
        slice(None, -15),
        slice(100, None),
    ]:
        with pytest.raises(IndexError):
            h[sl]

    # Reversed bounds are also empty under integration (but flow access like
    # h[3::sum] on a size-3 axis stays valid; see test_single_flow_bin).
    with pytest.raises(IndexError):
        h[5:2:sum]


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

    assert repr(bh.rebin(2)) == "rebin(factor=2)"


# Was broken in 0.6.1
def test_noflow_slicing():
    noflow = {"underflow": False, "overflow": False}

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

    assert h[:, :, True].view() == approx(vals)
    assert h[:, :, False].view() == approx(0)


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

    assert np.asarray(h[:, 1 : 3 : bh.sum]) == approx(vals[:, 1:3].sum(axis=1))
    assert np.asarray(h[{1: slice(1, 3, bh.sum)}]) == approx(vals[:, 1:3].sum(axis=1))
    assert np.asarray(h[1 : 3 : bh.sum, :]) == approx(vals[1:3, :].sum(axis=0))


def test_set_range_with_scalar():
    h = bh.Histogram(bh.axis.Integer(0, 10))
    h[2:5] = 42

    assert h[1] == 0
    assert h[2] == 42
    assert h[3] == 42
    assert h[4] == 42
    assert h[5] == 0


def test_set_range_with_scalar_callable():
    h = bh.Histogram(bh.axis.Integer(0, 10))
    h[2:len] = 42

    assert h[1] == 0
    assert h[2] == 42
    assert h[3] == 42
    assert h[4] == 42
    assert h[5] == 42
    assert h[bh.overflow] == 0


def test_set_all_with_scalar():
    h = bh.Histogram(bh.axis.Integer(0, 10))
    h[:] = 42

    assert h[0] == 42
    assert h[9] == 42
    assert h[::sum] == 42 * 10


def test_pick_str_category():
    noflow = {"underflow": False, "overflow": False}

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

    assert h[:, :, bh.loc("on")].view() == approx(vals)
    assert h[{2: bh.loc("on")}].view() == approx(vals)
    assert h[:, :, bh.loc("off")].view() == approx(0)


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
    noflow = {"underflow": False, "overflow": False}

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

    assert h[:, :, bh.loc(3)].view() == approx(vals)
    assert h[{2: bh.loc(3)}].view() == approx(vals)
    assert h[:, :, bh.loc(5)].view() == approx(vals + 1)
    assert h[:, :, bh.loc(7)].view() == approx(0)


@pytest.mark.parametrize(
    "ax",
    [bh.axis.Regular(3, 0, 1), bh.axis.Variable([0, 0.3, 0.6, 1])],
    ids=["regular", "variable"],
)
def test_pick_flowbin(ax):
    w = 1e-2  # e.g. a cross section for a process
    x = [-0.1, -0.1, 0.1, 0.1, 0.1]
    y = [-0.1, 0.1, -0.1, -0.1, 0.1]

    h = bh.Histogram(
        ax,
        ax,
        storage=bh.storage.Weight(),
    )
    h.fill(x, y, weight=w)

    uf_slice = h[bh.tag.underflow, ...]
    assert uf_slice.values(flow=True) == approx(np.array([1, 1, 0, 0, 0]) * w)

    uf_slice = h[..., bh.tag.underflow]
    assert uf_slice.values(flow=True) == approx(np.array([1, 2, 0, 0, 0]) * w)


def test_axes_tuple():
    h = bh.Histogram(bh.axis.Regular(10, 0, 1))
    assert isinstance(h.axes[:1], bh.axis.AxesTuple)
    assert isinstance(h.axes[0], bh.axis.Regular)

    (before,) = h.axes.centers[:1]
    (after,) = h.axes[:1].centers

    assert before == approx(after)


def test_axes_tuple_Nd():
    h = bh.Histogram(
        bh.axis.Integer(0, 5), bh.axis.Integer(0, 4), bh.axis.Integer(0, 6)
    )
    assert isinstance(h.axes[:2], bh.axis.AxesTuple)
    assert isinstance(h.axes[1], bh.axis.Integer)

    b1, b2 = h.axes.centers[1:3]
    a1, a2 = h.axes[1:3].centers

    assert b1.flatten() == approx(a1.flatten())
    assert b2.flatten() == approx(a2.flatten())

    assert b1.ndim == 3
    assert a1.ndim == 2


# issue 556
def test_single_flow_bin():
    # Flow is removed for category axes unless full sum is used
    h = bh.Histogram(bh.axis.IntCategory([0, 1, 2]))
    h.view(True)[:] = 1

    assert h[::sum] == 4
    assert h[0::sum] == 3
    assert h[1::sum] == 2
    assert h[2::sum] == 1
    # start past the last bin selects no bins -> clear empty-slice error
    with pytest.raises(IndexError):
        h[3::sum]

    assert h[1:2][sum] == 4

    h = bh.Histogram(bh.axis.Integer(0, 3))
    h.view(True)[:] = 1

    assert h[::sum] == 5
    assert h[0::sum] == 4
    assert h[1::sum] == 3
    assert h[2::sum] == 2
    assert h[3::sum] == 1

    assert h[1:2][sum] == 5


# issue 579


def test_scale_flowbins():
    w = 1e-1
    x = np.random.normal(loc=0.4, scale=0.4, size=100)

    h = bh.Histogram(bh.axis.Variable([0, 0.5, 1]), storage=bh.storage.Weight())

    h.fill(x, weight=w)

    ref_value = h.values(flow=True) * 5
    scale_value = (h * 5).values(flow=True)

    assert scale_value == approx(ref_value)


def test_add_flowbins():
    w = 1e-1
    x = np.random.normal(loc=0.4, scale=0.4, size=100)

    h = bh.Histogram(bh.axis.Variable([0, 0.5, 1]), storage=bh.storage.Weight())

    h.fill(x, weight=w)

    ref_value = h.values(flow=True) + 5
    scale_value = (h + 5).values(flow=True)

    assert scale_value == approx(ref_value)
    assert (scale_value != 5).any()


# issue 737
def test_large_index():
    h = bh.Histogram(
        bh.axis.IntCategory([4, 8, 15, 16, 23, 42, 99_999_001, 1_000_010_020])
    )
    assert h.axes[0].value(6) == 99_999_001
    assert h.axes[0].index(99_999_001) == 6


def test_scaling_slice():
    h = bh.Histogram(bh.axis.Regular(3, 0, 3), bh.axis.StrCategory(["a", "b"]))
    h.fill([1, 1, 2], "a")
    h.fill([0], "b")

    h[:, bh.loc("a")] *= 2

    assert h[1, 0] == approx(4)
    assert h[2, 0] == approx(2)
    assert h[0, 1] == approx(1)


def test_scaling_slice_weight():
    h = bh.Histogram(
        bh.axis.Regular(3, 0, 3),
        bh.axis.StrCategory(["a", "b"]),
        storage=bh.storage.Weight(),
    )
    h.fill([1, 1, 2], "a")
    h.fill([0], "b")

    h[:, bh.loc("a")] *= 2

    assert h[1, 0].value == approx(4)
    assert h[2, 0].value == approx(2)
    assert h[0, 1].value == approx(1)


def test_setting_histogram_mismatch():
    h = bh.Histogram(bh.axis.Regular(10, 0, 10), bh.axis.Integer(0, 20))

    h[:, 0] = bh.Histogram(bh.axis.Regular(10, 0, 10))
    h[0:, 0] = bh.Histogram(bh.axis.Regular(10, 0, 10, underflow=False))
    h[:len, 0] = bh.Histogram(bh.axis.Regular(10, 0, 10, overflow=False))
    h[0:len, 0] = bh.Histogram(
        bh.axis.Regular(10, 0, 10, underflow=False, overflow=False)
    )
    with pytest.raises(ValueError, match="Cannot set histogram with underflow"):
        h[0:, 0] = bh.Histogram(bh.axis.Regular(10, 0, 10))
    with pytest.raises(ValueError, match="Cannot set histogram with underflow"):
        h[:len, 0] = bh.Histogram(bh.axis.Regular(10, 0, 10))
    with pytest.raises(ValueError, match="Cannot set histogram with underflow"):
        h[:, 0] = bh.Histogram(bh.axis.Regular(10, 0, 10, underflow=False))
    with pytest.raises(ValueError, match="Cannot set histogram with underflow"):
        h[:, 0] = bh.Histogram(bh.axis.Regular(10, 0, 10, overflow=False))
    with pytest.raises(ValueError, match="Cannot set histogram with underflow"):
        h[:, 0] = bh.Histogram(
            bh.axis.Regular(10, 0, 10, underflow=False, overflow=False)
        )
    with pytest.raises(ValueError, match="Cannot set histogram with underflow"):
        h[0:, 0] = bh.Histogram(
            bh.axis.Regular(10, 0, 10, underflow=False, overflow=False)
        )
    with pytest.raises(ValueError, match="Cannot set histogram with underflow"):
        h[:len, 0] = bh.Histogram(
            bh.axis.Regular(10, 0, 10, underflow=False, overflow=False)
        )


def test_setting_histogram_slice_not_at_start():
    """Regression test: __setitem__ with slices that are not at the start of the index.

    Previously, h[2,:,:,2] = other_hist would use the wrong axis indices when
    checking underflow/overflow traits of the value histogram, causing either
    a ValueError (wrong axis compared) or IndexError (index out of range).
    """
    axis_wUnderflow0 = bh.axis.Regular(10, 0, 1, underflow=True, overflow=True)
    axis_wUnderflow1 = bh.axis.Regular(5, 0, 5, underflow=True, overflow=True)
    axis_wUnderflow2 = bh.axis.Regular(7, 0, 3, underflow=True, overflow=True)
    axis_noUnderflow = bh.axis.Regular(8, -2, 2, underflow=False, overflow=True)

    h_2D = bh.Histogram(axis_noUnderflow, axis_wUnderflow0)
    h_2D.fill([-1.1, -0.2, 0.3, 1.4], [0.2, 0.5, 0.3, 0.8])

    # Slices in the middle of the index
    h_middle = bh.Histogram(
        axis_wUnderflow1, axis_noUnderflow, axis_wUnderflow0, axis_wUnderflow2
    )
    h_middle[2, :, :, 2] = h_2D  # previously raised ValueError

    # Slices at the end of the index
    h_last = bh.Histogram(
        axis_wUnderflow1, axis_wUnderflow2, axis_noUnderflow, axis_wUnderflow0
    )
    h_last[2, 2, :, :] = h_2D  # previously raised IndexError

    # Slices at the start (sanity check - this always worked)
    h_first = bh.Histogram(
        axis_noUnderflow, axis_wUnderflow0, axis_wUnderflow1, axis_wUnderflow2
    )
    h_first[:, :, 2, 2] = h_2D

    # Verify that the data was correctly set (use a non-flow slice comparison)
    assert h_first.view()[:, :, 2, 2] == approx(h_2D.view())
    assert h_middle.view()[2, :, :, 2] == approx(h_2D.view())
    assert h_last.view()[2, 2, :, :] == approx(h_2D.view())


def test_rebin_groups_no_inplace_modification():
    """
    Test that rebinning with a groups list does not mutate the input list in-place.
    This ensures consecutive rebin operations with the same list do not fail.
    """
    h1 = bh.Histogram(bh.axis.Regular(60, 0, 600))
    h2 = bh.Histogram(bh.axis.Regular(60, 0, 600))
    rebinner = [5] * 12
    h3 = h1[bh.rebin(groups=rebinner)]
    # The original list should remain unchanged
    assert rebinner == [5] * 12, "rebinner list was mutated in-place"
    # Second rebin should not raise
    h4 = h2[bh.rebin(groups=rebinner)]
    assert h3 == h4


# Issue #1143 (B7)
def test_rebin_group_mapping_factor():
    # The factor form of the RebinProtocol must produce groups that sum to
    # the number of bins when the factor divides evenly
    ax = bh.axis.Regular(10, 0, 10)
    groups = bh.rebin(2).group_mapping(ax)
    assert list(groups) == [2] * 5
    assert sum(groups) == len(ax)

    # With a remainder, the leftover bins are not part of any group (they
    # are merged into overflow, matching the C++ factor rebinning)
    ax5 = bh.axis.Regular(5, 0, 5)
    assert list(bh.rebin(2).group_mapping(ax5)) == [2, 2]

    # The group form of group_mapping must match the C++ factor rebinning
    h = bh.Histogram(bh.axis.Regular(5, 0, 5))
    h.view(flow=True)[:] = [100, 1, 2, 3, 4, 5, 200]
    hs = h[:: bh.rebin(2)]
    assert hs.view(flow=True) == approx([100, 3, 7, 205])


def test_rebin_factor_and_axis_raises():
    with pytest.raises(ValueError, match="factor cannot be combined"):
        bh.rebin(2, axis=bh.axis.Regular(2, 0, 4))
    with pytest.raises(ValueError, match="factor cannot be combined"):
        bh.rebin(factor=2, axis=bh.axis.Regular(2, 0, 4))


def test_vectorized_get_basic():
    """NumPy integer-array indices gather scattered cells through the buffer."""
    h = bh.Histogram(
        bh.axis.IntCategory(list(range(5))),
        bh.axis.IntCategory(list(range(6))),
        bh.axis.Regular(7, 0, 7),
    )
    h.view()[...] = np.arange(5 * 6 * 7).reshape(5, 6, 7)

    i0 = np.array([0, 2, 4])
    i1 = np.array([1, 3, 5])
    i2 = np.array([2, 4, 6])

    # Full gather (no slice) -> 1D array of values, like view() fancy indexing
    assert h[i0, i1, i2] == approx(h.view()[i0, i1, i2])

    # Arrays plus a trailing slice keep the axis
    assert h[i0, i1, :] == approx(h.view()[i0, i1, :])

    # Arrays plus an ellipsis expand the remaining axes
    assert h[i0, i1, ...] == approx(h.view()[i0, i1, :])


def test_vectorized_get_mixed_scalar_and_loc():
    h = bh.Histogram(
        bh.axis.StrCategory([str(i) for i in range(4)]),
        bh.axis.StrCategory([str(i) for i in range(4)]),
        bh.axis.Regular(5, 0, 5),
    )
    h.view()[...] = np.arange(4 * 4 * 5).reshape(4, 4, 5)

    i1 = np.array([0, 2])
    # Scalar locator + array + slice mix
    assert h[bh.loc("1"), i1, :] == approx(h.view()[1, i1, :])
    assert h[2, i1, :] == approx(h.view()[2, i1, :])


def test_vectorized_get_flow_offset():
    """Array indices are non-flow, matching scalar __getitem__ semantics."""
    h = bh.Histogram(bh.axis.Regular(5, 0, 5))
    h.view(flow=True)[...] = np.arange(7.0)

    idx = np.array([0, 4])
    assert h[idx] == approx(np.array([h[0], h[4]]))


def test_vectorized_get_accumulator_storage():
    h = bh.Histogram(bh.axis.Regular(5, 0, 5), storage=bh.storage.Weight())
    h.fill([0, 0, 1, 2, 3], weight=[1, 2, 3, 4, 5])

    idx = np.array([0, 1, 3])
    got = h[idx]
    assert isinstance(got, bh.view.WeightedSumView)
    assert got.value == approx(h.view()[idx].value)
    assert got.variance == approx(h.view()[idx].variance)


def test_vectorized_get_multicell_storage():
    h = bh.Histogram(
        bh.axis.Regular(4, 0, 4),
        bh.axis.Regular(4, 0, 4),
        storage=bh.storage.MultiCell(3),
    )
    h.view()[...] = np.arange(np.prod(h.view().shape)).reshape(h.view().shape)

    i0 = np.array([1, 2])
    i1 = np.array([0, 3])
    # The cell index stays the leading dimension of the result
    assert h[i0, i1] == approx(h.view()[:, i0, i1])


def test_vectorized_get_rejects_unsupported():
    h = bh.Histogram(bh.axis.IntCategory([1, 2, 3]), bh.axis.Regular(5, 0, 5))
    arr = np.array([0, 1])

    with pytest.raises(IndexError, match="rebin, sum, or locator slices"):
        h[arr, :: bh.rebin(2)]

    with pytest.raises(IndexError, match="rebin, sum, or locator slices"):
        h[arr, ::sum]

    # A categorical pick list cannot be combined with array indexing
    with pytest.raises(IndexError, match="integer arrays"):
        h[[0, 1], arr]


def test_negative_size_index():
    # Issue #1143 (B8a): -size is a valid Python index for bin 0
    h = bh.Histogram(bh.axis.Regular(10, 0, 1))
    h[0] = 7

    assert h[-10] == 7
    assert h[-10] == h[0]

    with pytest.raises(IndexError):
        h[-11]
    with pytest.raises(IndexError):
        h[10]


def test_string_index_raises():
    # Issue #1143 (B14a): a plain string is not a valid index (use bh.loc)
    h = bh.Histogram(bh.axis.StrCategory(["a", "b"]))
    h.fill(["a", "a", "b"])

    with pytest.raises(IndexError, match="locator protocol"):
        h["a"]

    assert h[bh.loc("a")] == 2


def test_vectorized_get_set_respects_flow_locator():
    # A flow locator (bh.underflow/bh.overflow) mixed with an array index on
    # another axis must select the flow bin, not wrap into the regular range.
    h = bh.Histogram(bh.axis.Regular(4, 0, 1), bh.axis.Regular(3, 0, 1))
    h.view(flow=True)[...] = np.arange(6 * 5).reshape(6, 5)

    assert h[np.array([0]), bh.underflow] == approx([h.view(flow=True)[1, 0]])
    assert h[np.array([0, 1]), bh.overflow] == approx(
        [h.view(flow=True)[1, 4], h.view(flow=True)[2, 4]]
    )

    h[np.array([0]), bh.underflow] = 999
    assert h.view(flow=True)[1, 0] == 999


def test_vectorized_get_flow_locator_out_of_bounds_raises():
    h = bh.Histogram(
        bh.axis.Regular(4, 0, 1, underflow=False), bh.axis.Regular(3, 0, 1)
    )
    with pytest.raises(IndexError, match="no underflow bin"):
        h[bh.underflow, np.array([0])]


def test_underflow_locator_raises_without_underflow_bin():
    # Issue: bh.underflow on an axis without underflow silently wrapped to
    # the last (overflow) bin instead of raising.
    h = bh.Histogram(
        bh.axis.Regular(4, 0, 1, underflow=False), bh.axis.Regular(3, 0, 1)
    )
    h.view(flow=True)[...] = np.arange(5 * 5).reshape(5, 5)

    with pytest.raises(IndexError, match="no underflow bin"):
        h[bh.underflow, :]


def test_loc_below_range_raises_without_underflow_bin_setitem():
    # Issue: bh.loc(value_below_range) on an axis without underflow silently
    # wrote to the overflow bin instead of raising.
    h = bh.Histogram(bh.axis.Regular(4, 0, 1, underflow=False))

    with pytest.raises(IndexError, match="no underflow bin"):
        h[bh.loc(-5)] = 7

    assert h.view(flow=True) == approx([0, 0, 0, 0, 0])


def test_group_rebin_respects_slice_bounds():
    # Issue: h[start:stop:bh.rebin(groups=...)] ignored start/stop and
    # grouped the full axis instead of just the sliced range.
    h = bh.Histogram(bh.axis.Regular(6, 0, 6))
    h.view()[:] = [1, 2, 3, 4, 5, 6]

    hs = h[1 : 5 : bh.rebin(groups=[2, 2])]
    assert hs.view() == approx([2 + 3, 4 + 5])
    assert hs.axes[0].edges == approx([1, 3, 5])


def test_group_rebin_mismatched_sum_raises():
    h = bh.Histogram(bh.axis.Regular(6, 0, 6))
    h.view()[:] = [1, 2, 3, 4, 5, 6]

    with pytest.raises(ValueError, match="sum of the groups"):
        h[1 : 5 : bh.rebin(groups=[2, 3])]


def test_group_rebin_categorical_without_axis_raises():
    # Issue: group rebin on a category axis without axis= silently produced
    # a nonsensical Variable axis instead of a category axis.
    h = bh.Histogram(bh.axis.IntCategory([1, 2, 3, 4]))
    h.view()[:] = [10, 20, 30, 40]

    with pytest.raises(ValueError, match="categorical axis"):
        h[:: bh.rebin(groups=[2, 2])]


def test_group_rebin_categorical_with_axis():
    h = bh.Histogram(bh.axis.IntCategory([1, 2, 3, 4]))
    h.view()[:] = [10, 20, 30, 40]

    new_axis = bh.axis.IntCategory([1, 3])
    hs = h[:: bh.rebin(groups=[2, 2], axis=new_axis)]
    assert hs.view() == approx([30, 70])
    assert hs.axes[0] == new_axis


def test_rebin_numpy_integer_factor():
    # Issue: bh.rebin(np.int64(2)) was treated as an axis (isinstance(x, int)
    # is False for NumPy integer scalars), not a factor.
    r = bh.rebin(np.int64(2))
    assert r.factor == 2
    assert r.axis is None

    h = bh.Histogram(bh.axis.Regular(6, 0, 6))
    h.view()[:] = [1, 2, 3, 4, 5, 6]
    hs = h[:: bh.rebin(np.int64(2))]
    assert hs.view() == approx([3, 7, 11])


def test_rebin_bool_is_not_a_factor():
    with pytest.raises(TypeError, match="not a bool"):
        bh.rebin(True)


def test_reduce_command_repr():
    assert (
        repr(_core.algorithm.slice(0, 1, 3, _core.algorithm.slice_mode.crop))
        == "reduce_command(slice(iaxis=0, begin=1, end=3, mode=slice_mode.crop))"
    )
    assert repr(_core.algorithm.crop(0, 1.0, 3.0)) == (
        "reduce_command(crop(iaxis=0, lower=1.0, upper=3.0))"
    )
    assert repr(_core.algorithm.shrink(0, 1.0, 3.0)) == (
        "reduce_command(shrink(iaxis=0, lower=1.0, upper=3.0))"
    )
    assert repr(_core.algorithm.shrink_and_rebin(0, 1.0, 3.0, 2)) == (
        "reduce_command(shrink_and_rebin(iaxis=0, lower=1.0, upper=3.0, merge=2))"
    )
