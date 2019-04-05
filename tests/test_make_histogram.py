import pytest
from pytest import approx
import math

import boost.histogram as bh

@pytest.mark.parametrize("axis,extent", ((bh.axis.regular_uoflow, 2),
                                         (bh.axis.regular_noflow, 0)))
def test_make_regular_1D(axis, extent):
    hist = bh.make_histogram(axis(3,2,5))

    assert hist.rank() == 1
    assert hist.axis(0).size() == 3
    assert hist.axis(0).size(flow=True) == 3 + extent
    assert hist.axis(0).bin(1).center() == approx(3.5)

@pytest.mark.parametrize("axis,extent", ((bh.axis.regular_uoflow, 2),
                                         (bh.axis.regular_noflow, 0)))
def test_make_regular_2D(axis, extent):
    hist = bh.make_histogram(axis(3,2,5),
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
    hist = bh.make_histogram(bh.axis.regular_uoflow(5,1,2),
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

    assert float != type(bh.make_histogram(bh.axis.regular_uoflow(5,1,2), storage=bh.storage.int()).at(0))
    assert float == type(bh.make_histogram(bh.axis.regular_uoflow(5,1,2), storage=bh.storage.double()).at(0))

def test_issue_axis_bin_swan():
    hist = bh.make_histogram(bh.axis.regular_sqrt(10,0,10, metadata='x'),
                             bh.axis.circular(10,0,1, metadata='y'))

    b = hist.axis(1).bin(1)
    assert repr(b) == '<bin [0.100000, 0.200000]>'
    assert b.lower() == approx(0.1)
    assert b.upper() == approx(0.2)

    assert hist.axis(0).bin(0).lower() == 0
    assert hist.axis(0).bin(1).lower() == approx(.1)
    assert hist.axis(0).bin(2).lower() == approx(.4)
