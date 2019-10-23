import pytest
from pytest import approx

import boost_histogram as bh


@pytest.mark.parametrize("opt,extent", (("uo", 2), ("", 0)))
def test_make_regular_1D(opt, extent):
    hist = bh.histogram(
        bh.axis.regular(3, 2, 5, underflow="u" in opt, overflow="o" in opt)
    )

    assert hist.rank == 1
    assert hist.axes[0].size == 3
    assert hist.axes[0].extent == 3 + extent
    assert hist.axes[0].bin(1) == (3, 4)


def test_shortcuts():
    hist = bh.histogram((1, 2, 3), (10, 0, 1))
    assert hist.rank == 2
    for i in range(2):
        assert isinstance(hist.axes[i], bh.axis.regular)
        assert isinstance(hist.axes[i], bh.core.axis._regular_uoflow)
        assert not isinstance(hist.axes[i], bh.axis.variable)


def test_shortcuts_with_metadata():
    bh.histogram((1, 2, 3, "this"))
    bh.histogram((1, 2, 3, 4))
    with pytest.raises(TypeError):
        bh.histogram((1, 2))
    with pytest.raises(TypeError):
        bh.histogram((1, 2, 3, 4, 5))


@pytest.mark.parametrize("opt,extent", (("uo", 2), ("", 0)))
def test_make_regular_2D(opt, extent):
    hist = bh.histogram(
        bh.axis.regular(3, 2, 5, underflow="u" in opt, overflow="o" in opt),
        bh.axis.regular(5, 1, 6, underflow="u" in opt, overflow="o" in opt),
    )

    assert hist.rank == 2
    assert hist.axes[0].size == 3
    assert hist.axes[0].extent == 3 + extent
    assert hist.axes[0].bin(1) == approx((3, 4))

    assert hist.axes[1].size == 5
    assert hist.axes[1].extent == 5 + extent
    assert hist.axes[1].bin(1) == approx((2, 3))


@pytest.mark.parametrize(
    "storage",
    (
        bh.storage.int(),
        bh.storage.double(),
        bh.storage.unlimited(),
        bh.storage.weight(),
    ),
)
def test_make_any_hist(storage):
    hist = bh.histogram(
        bh.axis.regular(3, 1, 4),
        bh.axis.regular(2, 2, 4, underflow=False, overflow=False),
        bh.axis.circular(4, 1, 5),
        storage=storage,
    )

    assert hist.rank == 3
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

    assert float != type(
        bh.histogram(bh.axis.regular(5, 1, 2), storage=bh.storage.int())[0]
    )
    assert float == type(
        bh.histogram(bh.axis.regular(5, 1, 2), storage=bh.storage.double())[0]
    )


def test_issue_axis_bin_swan():
    hist = bh.histogram(
        bh.axis.regular_sqrt(10, 0, 10, metadata="x"),
        bh.axis.circular(10, 0, 1, metadata="y"),
    )

    b = hist.axes[1].bin(1)
    assert repr(b) == "(0.1, 0.2)"
    assert b[0] == approx(0.1)
    assert b[1] == approx(0.2)

    assert hist.axes[0].bin(0)[0] == 0
    assert hist.axes[0].bin(1)[0] == approx(0.1)
    assert hist.axes[0].bin(2)[0] == approx(0.4)


options = (
    (
        bh.core.hist._any_unlimited,
        bh.core.axis._regular_uoflow(5, 1, 2, None),
        bh.storage.unlimited,
    ),
    (
        bh.core.hist._any_int,
        bh.core.axis._regular_uoflow(5, 1, 2, None),
        bh.storage.int,
    ),
    (
        bh.core.hist._any_atomic_int,
        bh.core.axis._regular_uoflow(5, 1, 2, None),
        bh.storage.atomic_int,
    ),
    (bh.core.hist._any_int, bh.core.axis._regular_none(5, 1, 2, None), bh.storage.int),
    (
        bh.core.hist._any_double,
        bh.core.axis._regular_uoflow(5, 1, 2, None),
        bh.storage.double,
    ),
    (
        bh.core.hist._any_weight,
        bh.core.axis._regular_uoflow(5, 1, 2, None),
        bh.storage.weight,
    ),
    (bh.core.hist._any_int, bh.core.axis._integer_uoflow(0, 5, None), bh.storage.int),
    (
        bh.core.hist._any_atomic_int,
        bh.core.axis._integer_uoflow(0, 5, None),
        bh.storage.atomic_int,
    ),
    (
        bh.core.hist._any_double,
        bh.core.axis._integer_uoflow(0, 5, None),
        bh.storage.double,
    ),
    (
        bh.core.hist._any_unlimited,
        bh.core.axis._integer_uoflow(0, 5, None),
        bh.storage.unlimited,
    ),
    (
        bh.core.hist._any_weight,
        bh.core.axis._integer_uoflow(0, 5, None),
        bh.storage.weight,
    ),
)


@pytest.mark.parametrize("histclass, ax, storage", options)
def test_make_selection(histclass, ax, storage):
    histogram = bh.histogram(ax, storage=storage())
    assert isinstance(histogram, histclass)

    histogram = bh.histogram(ax, ax, storage=storage())
    assert isinstance(histogram, histclass)


def test_make_selection_special():
    histogram = bh.histogram(
        bh.axis.regular(5, 1, 2),
        bh.axis.regular(10, 1, 2, underflow=False, overflow=False),
    )
    assert isinstance(histogram, bh.core.hist._any_double)
