import pytest
from pytest import approx
import math

import boost_histogram as bh


@pytest.mark.parametrize(
    "axis,extent",
    (
        (bh.core.axis._regular_uoflow, 2),
        (lambda *x: x, 2),
        (bh.core.axis._regular_noflow, 0),
    ),
)
def test_make_regular_1D(axis, extent):
    hist = bh.histogram(axis(3, 2, 5))

    assert hist.rank == 1
    assert hist.axis(0).size == 3
    assert hist.axis(0).extent == 3 + extent
    assert hist.axis(0).bin(1) == (3, 4)


def test_shortcuts():
    hist = bh.histogram((1, 2, 3), (10, 0, 1))
    assert hist.rank == 2
    for i in range(2):
        assert isinstance(hist.axis(i), bh.axis.regular)
        assert isinstance(hist.axis(i), bh.core.axis._regular_uoflow)
        assert not isinstance(hist.axis(i), bh.axis.variable)


def test_shortcuts_with_metadata():
    bh.histogram((1, 2, 3, "this"))
    with pytest.raises(TypeError):
        bh.histogram((1, 2, 3, 4))
    with pytest.raises(TypeError):
        bh.histogram((1, 2))
    with pytest.raises(TypeError):
        bh.histogram((1, 2, 3, 4, 5))


@pytest.mark.parametrize(
    "axis,extent",
    ((bh.core.axis._regular_uoflow, 2), (bh.core.axis._regular_noflow, 0)),
)
def test_make_regular_2D(axis, extent):
    hist = bh.histogram(axis(3, 2, 5), axis(5, 1, 6))

    assert hist.rank == 2
    assert hist.axis(0).size == 3
    assert hist.axis(0).extent == 3 + extent
    assert hist.axis(0).bin(1) == approx((3, 4))

    assert hist.axis(1).size == 5
    assert hist.axis(1).extent == 5 + extent
    assert hist.axis(1).bin(1) == approx((2, 3))


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
        bh.core.axis._regular_uoflow(3, 1, 4),
        bh.core.axis._regular_noflow(2, 2, 4),
        bh.axis.circular(4, 1, 5),
        storage=storage,
    )

    assert hist.rank == 3
    assert hist.axis(0).size == 3
    assert hist.axis(0).extent == 5
    assert hist.axis(0).bin(1) == approx((2, 3))
    assert hist.axis(1).size == 2
    assert hist.axis(1).extent == 2
    assert hist.axis(1).bin(1) == approx((3, 4))
    assert hist.axis(2).size == 4
    assert hist.axis(2).extent == 5
    assert hist.axis(2).bin(1) == approx((2, 3))


def test_make_any_hist_storage():

    assert float != type(
        bh.histogram(
            bh.core.axis._regular_uoflow(5, 1, 2), storage=bh.storage.int()
        ).at(0)
    )
    assert float == type(
        bh.histogram(
            bh.core.axis._regular_uoflow(5, 1, 2), storage=bh.storage.double()
        ).at(0)
    )


def test_issue_axis_bin_swan():
    hist = bh.histogram(
        bh.axis.regular_sqrt(10, 0, 10, metadata="x"),
        bh.axis.circular(10, 0, 1, metadata="y"),
    )

    b = hist.axis(1).bin(1)
    assert repr(b) == "(0.1, 0.2)"
    assert b[0] == approx(0.1)
    assert b[1] == approx(0.2)

    assert hist.axis(0).bin(0)[0] == 0
    assert hist.axis(0).bin(1)[0] == approx(0.1)
    assert hist.axis(0).bin(2)[0] == approx(0.4)


options = (
    (
        bh.core.hist._any_unlimited,
        bh.core.axis._regular_uoflow(5, 1, 2),
        bh.storage.unlimited,
    ),
    (bh.core.hist._any_int, bh.core.axis._regular_uoflow(5, 1, 2), bh.storage.int),
    (
        bh.core.hist._any_atomic_int,
        bh.core.axis._regular_uoflow(5, 1, 2),
        bh.storage.atomic_int,
    ),
    (bh.core.hist._any_int, bh.core.axis._regular_noflow(5, 1, 2), bh.storage.int),
    (
        bh.core.hist._any_double,
        bh.core.axis._regular_uoflow(5, 1, 2),
        bh.storage.double,
    ),
    (
        bh.core.hist._any_weight,
        bh.core.axis._regular_uoflow(5, 1, 2),
        bh.storage.weight,
    ),
    (bh.core.hist._any_int, bh.core.axis._integer_uoflow(0, 5), bh.storage.int),
    (
        bh.core.hist._any_atomic_int,
        bh.core.axis._integer_uoflow(0, 5),
        bh.storage.atomic_int,
    ),
    (bh.core.hist._any_double, bh.core.axis._integer_uoflow(0, 5), bh.storage.double),
    (
        bh.core.hist._any_unlimited,
        bh.core.axis._integer_uoflow(0, 5),
        bh.storage.unlimited,
    ),
    (bh.core.hist._any_weight, bh.core.axis._integer_uoflow(0, 5), bh.storage.weight),
)


@pytest.mark.parametrize("histclass, ax, storage", options)
def test_make_selection(histclass, ax, storage):
    histogram = bh.histogram(ax, storage=storage())
    assert isinstance(histogram, histclass)

    histogram = bh.histogram(ax, ax, storage=storage())
    assert isinstance(histogram, histclass)


def test_make_selection_special():
    histogram = bh.histogram(
        bh.core.axis._regular_uoflow(5, 1, 2), bh.core.axis._regular_noflow(10, 1, 2)
    )
    assert isinstance(histogram, bh.core.hist._any_double)
