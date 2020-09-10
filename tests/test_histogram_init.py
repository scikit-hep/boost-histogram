# -*- coding: utf-8 -*-
import pytest
from pytest import approx

import boost_histogram as bh


@pytest.mark.parametrize("opt,extent", (("uo", 2), ("", 0)))
def test_make_regular_1D(opt, extent):
    hist = bh.Histogram(
        bh.axis.Regular(3, 2, 5, underflow="u" in opt, overflow="o" in opt)
    )

    assert hist.ndim == 1
    assert hist.axes[0].size == 3
    assert hist.axes[0].extent == 3 + extent
    assert hist.axes[0].bin(1) == (3, 4)


@pytest.mark.filterwarnings("ignore")
def test_shortcuts():
    hist = bh.Histogram((1, 2, 3), (10, 0, 1))
    assert hist.ndim == 2
    for i in range(2):
        assert isinstance(hist.axes[i], bh.axis.Regular)
        assert not isinstance(hist.axes[i], bh.axis.Variable)


@pytest.mark.filterwarnings("ignore")
def test_shortcuts_with_metadata():
    with pytest.raises(TypeError):
        bh.Histogram((1, 2, 3, 4))
    with pytest.raises(TypeError):
        bh.Histogram((1, 2))
    with pytest.raises(TypeError):
        bh.Histogram((1, 2, 3, 4, 5))
    with pytest.raises(TypeError):
        bh.Histogram((1, 2, 3, "this"))


@pytest.mark.parametrize("opt,extent", (("uo", 2), ("", 0)))
def test_make_regular_2D(opt, extent):
    hist = bh.Histogram(
        bh.axis.Regular(3, 2, 5, underflow="u" in opt, overflow="o" in opt),
        bh.axis.Regular(5, 1, 6, underflow="u" in opt, overflow="o" in opt),
    )

    assert hist.ndim == 2
    assert hist.axes[0].size == 3
    assert hist.axes[0].extent == 3 + extent
    assert hist.axes[0].bin(1) == approx((3, 4))

    assert hist.axes[1].size == 5
    assert hist.axes[1].extent == 5 + extent
    assert hist.axes[1].bin(1) == approx((2, 3))


@pytest.mark.parametrize(
    "storage",
    (
        bh.storage.Int64(),
        bh.storage.Double(),
        bh.storage.Unlimited(),
        bh.storage.Weight(),
    ),
)
def test_make_any_hist(storage):
    hist = bh.Histogram(
        bh.axis.Regular(3, 1, 4),
        bh.axis.Regular(2, 2, 4, underflow=False, overflow=False),
        bh.axis.Regular(4, 1, 5, circular=True),
        storage=storage,
    )

    assert hist.ndim == 3
    assert hist.axes[0].size == 3
    assert hist.axes[0].extent == 5
    assert hist.axes[0].bin(1) == approx((2, 3))
    assert hist.axes[1].size == 2
    assert hist.axes[1].extent == 2
    assert hist.axes[1].bin(1) == approx((3, 4))
    assert hist.axes[2].size == 4
    assert hist.axes[2].extent == 5
    assert hist.axes[2].bin(1) == approx((2, 3))


def test_make_any_hist_storage():

    assert not isinstance(
        bh.Histogram(bh.axis.Regular(5, 1, 2), storage=bh.storage.Int64())[0], float
    )
    assert isinstance(
        bh.Histogram(bh.axis.Regular(5, 1, 2), storage=bh.storage.Double())[0], float
    )
    assert isinstance(bh.Histogram(bh.axis.Regular(5, 1, 2), storage=None)[0], float)


def test_issue_axis_bin_swan():
    hist = bh.Histogram(
        bh.axis.Regular(10, 0, 10, metadata="x", transform=bh.axis.transform.sqrt),
        bh.axis.Regular(10, 0, 1, metadata="y", circular=True),
    )

    b = hist.axes[1].bin(1)
    assert repr(b) == "(0.1, 0.2)"
    assert b[0] == approx(0.1)
    assert b[1] == approx(0.2)

    assert hist.axes[0].bin(0)[0] == 0
    assert hist.axes[0].bin(1)[0] == approx(0.1)
    assert hist.axes[0].bin(2)[0] == approx(0.4)


hist_ax = (
    bh.axis.Regular(5, 1, 2, metadata=None),
    bh.axis.Regular(5, 1, 2, metadata=None, overflow=False, underflow=False),
    bh.axis.Integer(0, 5, metadata=None),
)
hist_storage = (
    bh.storage.Double,
    bh.storage.Unlimited,
    bh.storage.Int64,
    bh.storage.AtomicInt64,
    bh.storage.Weight,
)


@pytest.mark.parametrize("ax", hist_ax)
@pytest.mark.parametrize("storage", hist_storage)
def test_make_selection(ax, storage):
    histogram = bh.Histogram(ax, storage=storage())
    assert isinstance(histogram, bh.Histogram)

    histogram = bh.Histogram(ax, ax, storage=storage())
    assert isinstance(histogram, bh.Histogram)
    # TODO: Make this test do something useful
