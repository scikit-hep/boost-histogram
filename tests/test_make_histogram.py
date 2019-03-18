import pytest
from pytest import approx

import histogram as bh

@pytest.mark.parametrize("axis,extent", ((bh.axis.regular, 2),
                                         (bh.axis.regular_noflow, 0)))
def test_make_regular_1D(axis, extent):
    hist = bh.make_histogram(axis(3,2,5))

    assert hist.rank() == 1
    assert hist.axis(0).size() == 3
    assert hist.axis(0).extent() == 3 + extent
    assert hist.axis(0).bin(1).center() == approx(3.5)

@pytest.mark.parametrize("axis,extent", ((bh.axis.regular, 2),
                                         (bh.axis.regular_noflow, 0)))
def test_make_regular_2D(axis, extent):
    hist = bh.make_histogram(axis(3,2,5),
                             axis(5,1,6))

    assert hist.rank() == 2
    assert hist.axis(0).size() == 3
    assert hist.axis(0).extent() == 3 + extent
    assert hist.axis(0).bin(1).center() == approx(3.5)

    assert hist.axis(1).size() == 5
    assert hist.axis(1).extent() == 5 + extent
    assert hist.axis(1).bin(1).center() == approx(2.5)


@pytest.mark.parametrize("storage", (bh.storage.dense_int(),
                                     bh.storage.dense_double(),
                                     bh.storage.unlimited(),
                                     bh.storage.weight()))
def test_make_any_hist(storage):
    hist = bh.make_histogram(bh.axis.regular(5,1,2),
                             bh.axis.regular_noflow(6,2,3),
                             bh.axis.circular(7,3,4),
                             storage=storage)

    assert hist.rank() == 3
    assert hist.axis(0).size() == 5
    assert hist.axis(0).extent() == 7
    assert hist.axis(1).size() == 6
    assert hist.axis(1).extent() == 6
    assert hist.axis(2).size() == 7
    assert hist.axis(2).extent() == 8

def test_make_any_hist_storage():

    assert float != type(bh.make_histogram(bh.axis.regular(5,1,2), storage=bh.storage.dense_int()).at(0))
    assert float == type(bh.make_histogram(bh.axis.regular(5,1,2), storage=bh.storage.dense_double()).at(0))
