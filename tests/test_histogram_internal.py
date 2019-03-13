import pytest

import histogram as bh
import numpy as np


def test_1D_fill_unlimited():
    bins = 10
    ranges = (0, 1)
    vals = (.15, .25, .25)
    hist = bh.hist.regular_unlimited([
        bh.axis.regular(bins, *ranges)
        ])
    hist(vals)


methods = [
    bh.hist.regular_int,
    bh.hist.any_int,
]

@pytest.mark.parametrize("hist_func", methods + [bh.hist.regular_int_1d])
def test_1D_fill_int(hist_func):
    bins = 10
    ranges = (0, 1)

    vals = (.15, .25, .25)

    hist = hist_func([
        bh.axis.regular(bins, *ranges)
        ])
    hist(vals)

    H =  np.array([0, 1, 2, 0, 0, 0, 0, 0, 0, 0])

    assert np.all(np.asarray(hist)[1:-1] == H)

@pytest.mark.parametrize("hist_func", methods + [bh.hist.regular_int_2d])
def test_2D_fill_int(hist_func):
    bins = (10, 10)
    ranges = ((0, 1), (0, 1))

    vals = ((.15, .25, .25), (.35, .45, .45))

    hist = hist_func([
        bh.axis.regular(bins[0], *ranges[0]),
        bh.axis.regular(bins[1], *ranges[1]),
        ])
    hist(*vals)

    H = np.histogram2d(*vals, bins=bins, range=ranges)[0]

    assert np.all(np.asarray(hist)[1:-1,1:-1] == H)


def test_edges_histogram():
    edges = (1, 12, 22, 79)
    hist = bh.hist.any_int([
        bh.axis.variable(edges)
        ])

    vals = (13,15,24,29)
    hist(vals)

    bins = np.asarray(hist)
    assert np.all(bins == [0,0,2,2,0])

def test_int_histogram():
    hist = bh.hist.any_int([
        bh.axis.integer(3,7)
        ])

    vals = (1,2,3,4,5,6,7,8,9)
    hist(vals)

    bins = np.asarray(hist)
    assert np.all(bins == [2,1,1,1,1,3])


def test_str_categories_histogram():
    hist = bh.hist.any_int([
        bh.axis.category_str(["a", "b", "c"])
        ])

    vals = ['a', 'b', 'b', 'c']
    # Can't fill yet

@pytest.mark.parametrize("axis", [
    bh.axis.regular,
    bh.axis.regular_noflow,
    bh.axis.circular,
    bh.axis.regular_log,
    bh.axis.regular_sqrt,
])
def test_regular_axis_repr(axis):
    ax = axis(2,3,4)
    assert 'object at' not in repr(ax)
