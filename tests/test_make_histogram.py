import pytest
from pytest import approx
import math

import boost.histogram as bh

@pytest.mark.parametrize("axis,extent", ((bh.axis.regular_uoflow, 2),
                                         (lambda *x: x, 2),
                                         (bh.axis.regular_noflow, 0)))
def test_make_regular_1D(axis, extent):
    hist = bh._make_histogram(axis(3,2,5))

    assert hist.rank() == 1
    assert hist.axis(0).size() == 3
    assert hist.axis(0).size(flow=True) == 3 + extent
    assert hist.axis(0).bin(1).center() == approx(3.5)

def test_shortcuts():
    hist = bh.histogram((1,2,3), (10,0,1))
    assert hist.rank() == 2
    for i in range(2):
        assert isinstance(hist.axis(i), bh.axis.regular)
        assert isinstance(hist.axis(i), bh.axis.regular_uoflow)
        assert not isinstance(hist.axis(i), bh.axis.variable)


def test_shortcuts_with_metadata():
    bh.histogram((1,2,3, "this"))
    with pytest.raises(TypeError):
        bh.histogram((1,2,3,4))
    with pytest.raises(TypeError):
        bh.histogram((1,2))
    with pytest.raises(TypeError):
        bh.histogram((1,2,3,4,5))



@pytest.mark.parametrize("axis,extent", ((bh.axis.regular_uoflow, 2),
                                         (bh.axis.regular_noflow, 0)))
def test_make_regular_2D(axis, extent):
    hist = bh._make_histogram(axis(3,2,5),
                             axis(5,1,6))

    assert hist.rank() == 2
    assert hist.axis(0).size() == 3
    assert hist.axis(0).size(flow=True) == 3 + extent
    assert hist.axis(0).bin(1).center() == approx(3.5)

    assert hist.axis(1).size() == 5
    assert hist.axis(1).size(flow=True) == 5 + extent
    assert hist.axis(1).bin(1).center() == approx(2.5)


@pytest.mark.parametrize("storage", (bh.storage.int(),
                                     bh.storage.double(),
                                     bh.storage.unlimited(),
                                     bh.storage.weight()))
def test_make_any_hist(storage):
    hist = bh._make_histogram(bh.axis.regular_uoflow(5,1,2),
                             bh.axis.regular_noflow(6,2,3),
                             bh.axis.circular(8,3,4),
                             storage=storage)

    assert hist.rank() == 3
    assert hist.axis(0).size() == 5
    assert hist.axis(0).size(flow=True) == 7
    assert hist.axis(0).bin(1).center() == approx(1.3)
    assert hist.axis(1).size() == 6
    assert hist.axis(1).size(flow=True) == 6
    assert hist.axis(1).bin(1).center() == approx(2.25)
    assert hist.axis(2).size() == 8
    assert hist.axis(2).size(flow=True) == 9
    assert hist.axis(2).bin(1).center() == approx(3.1875)

def test_make_any_hist_storage():

    assert float != type(bh._make_histogram(bh.axis.regular_uoflow(5,1,2), storage=bh.storage.int()).at(0))
    assert float == type(bh._make_histogram(bh.axis.regular_uoflow(5,1,2), storage=bh.storage.double()).at(0))

def test_issue_axis_bin_swan():
    hist = bh._make_histogram(bh.axis.regular_sqrt(10,0,10, metadata='x'),
                             bh.axis.circular(10,0,1, metadata='y'))

    b = hist.axis(1).bin(1)
    assert repr(b) == '<bin [0.100000, 0.200000]>'
    assert b.lower() == approx(0.1)
    assert b.upper() == approx(0.2)

    assert hist.axis(0).bin(0).lower() == 0
    assert hist.axis(0).bin(1).lower() == approx(.1)
    assert hist.axis(0).bin(2).lower() == approx(.4)


options = (
        (bh.hist.regular_unlimited, bh.axis.regular_uoflow(5,1,2), bh.storage.unlimited),
        (bh.hist.regular_int, bh.axis.regular_uoflow(5,1,2), bh.storage.int),
        (bh.hist.regular_atomic_int, bh.axis.regular_uoflow(5,1,2), bh.storage.atomic_int),
        (bh.hist.regular_noflow_int, bh.axis.regular_noflow(5,1,2), bh.storage.int),
        (bh.hist.any_double, bh.axis.regular_uoflow(5,1,2), bh.storage.double),
        (bh.hist.any_weight, bh.axis.regular_uoflow(5,1,2), bh.storage.weight),
        (bh.hist.any_int, bh.axis.integer_uoflow(0,5), bh.storage.int),
        (bh.hist.any_atomic_int, bh.axis.integer_uoflow(0,5), bh.storage.atomic_int),
        (bh.hist.any_double, bh.axis.integer_uoflow(0,5), bh.storage.double),
        (bh.hist.any_unlimited, bh.axis.integer_uoflow(0,5), bh.storage.unlimited),
        (bh.hist.any_weight, bh.axis.integer_uoflow(0,5), bh.storage.weight),
        )

@pytest.mark.parametrize("histclass, ax, storage", options)
def test_make_selection(histclass, ax, storage):
    histogram = bh._make_histogram(ax, storage=storage())
    assert isinstance(histogram, histclass)

    histogram = bh._make_histogram(ax, ax, storage=storage())
    assert isinstance(histogram, histclass)


def test_make_selection_special():
    histogram = bh._make_histogram(bh.axis.regular_uoflow(5,1,2), bh.axis.regular_noflow(10,1,2))
    assert isinstance(histogram, bh.hist.any_int)

