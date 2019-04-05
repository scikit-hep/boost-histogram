import pytest

import boost.histogram as bh
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose


def test_1D_fill_unlimited():
    bins = 10
    ranges = (0, 1)
    vals = (.15, .25, .25)
    hist = bh.hist.regular_unlimited([
        bh.axis.regular_uoflow(bins, *ranges)
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
        bh.axis.regular_uoflow(bins, *ranges)
        ])
    hist(vals)

    H =  np.array([0, 1, 2, 0, 0, 0, 0, 0, 0, 0])

    assert_array_equal(np.asarray(hist)[1:-1], H)
    assert_array_equal(hist.view(flow=False), H)

    assert hist.axis(0).size() == bins
    assert hist.axis(0).size(flow=True) == bins + 2

@pytest.mark.parametrize("hist_func", methods + [bh.hist.regular_int_2d])
def test_2D_fill_int(hist_func):
    bins = (10, 15)
    ranges = ((0, 3), (0, 2))

    vals = ((.15, .25, .25), (.35, .45, .45))

    hist = hist_func([
        bh.axis.regular_uoflow(bins[0], *ranges[0]),
        bh.axis.regular_uoflow(bins[1], *ranges[1]),
        ])
    hist(*vals)

    H = np.histogram2d(*vals, bins=bins, range=ranges)[0]

    assert_array_equal(np.asarray(hist)[1:-1, 1:-1], H)
    assert_array_equal(hist.view(flow=False), H)

    assert hist.axis(0).size() == bins[0]
    assert hist.axis(0).size(flow=True) == bins[0] + 2

    assert hist.axis(1).size() == bins[1]
    assert hist.axis(1).size(flow=True) == bins[1] + 2


def test_edges_histogram():
    edges = (1, 12, 22, 79)
    hist = bh.hist.any_int([
        bh.axis.variable(edges)
        ])

    vals = (13,15,24,29)
    hist(vals)

    bins = np.asarray(hist)
    assert_array_equal(bins, [0,0,2,2,0])
    assert_array_equal(hist.view(flow=True), [0,0,2,2,0])
    assert_array_equal(hist.view(flow=False), [0,2,2])

def test_int_histogram():
    hist = bh.hist.any_int([
        bh.axis.integer_uoflow(3,7)
        ])

    vals = (1,2,3,4,5,6,7,8,9)
    hist(vals)

    bins = np.asarray(hist)
    assert_array_equal(bins, [2,1,1,1,1,3])
    assert_array_equal(hist.view(flow=True), [2,1,1,1,1,3])


def test_str_categories_histogram():
    hist = bh.hist.any_int([
        bh.axis.category_str(["a", "b", "c"])
        ])

    vals = ['a', 'b', 'b', 'c']
    # Can't fill yet

def test_growing_histogram():
    hist = bh.hist.any_int([
        bh.axis.regular_growth(10,0,1)
        ])

    hist(1.45)

    assert hist.size() == 15

def test_numpy_flow():
    h = bh.hist.regular_int_2d([bh.axis.regular_uoflow(10,0,1), bh.axis.regular_uoflow(5,0,1)])

    for i in range(10):
        for j in range(5):
            x,y = h.axis(0).bin(i).center(), h.axis(1).bin(j).center()
            v = i + j*10 + 1;
            h([x]*v,[y]*v)

    flow_true = h.to_numpy(True)[0][1:-1, 1:-1]
    flow_false = h.to_numpy(False)[0]

    assert_array_equal(flow_true, flow_false)

    view_flow_true = h.view(flow=True)
    view_flow_false = h.view(flow=False)
    view_flow_default = h.view()

    assert_array_equal(view_flow_true[1:-1, 1:-1], view_flow_false)
    assert_array_equal(view_flow_default, view_flow_false)



def test_numpy_compare():
    h = bh.hist.regular_int_2d([bh.axis.regular_uoflow(10,0,1), bh.axis.regular_uoflow(5,0,1)])

    xs = []
    ys = []
    for i in range(10):
        for j in range(5):
            x,y = h.axis(0).bin(i).center(), h.axis(1).bin(j).center()
            v = i + j*10 + 1;
            xs += [x]*v
            ys += [y]*v

    h(xs, ys)

    H, E1, E2 = h.to_numpy()

    nH, nE1, nE2 = np.histogram2d(xs, ys, bins=(10,5), range=((0,1),(0,1)))

    assert_array_equal(H, nH)
    assert_allclose(E1, nE1)
    assert_allclose(E2, nE2)

def test_int_cat_hist():
    h = bh.hist.any_int([bh.axis.category_int([1,2,3])])

    h(1)
    h(2)
    h(2.2)
    h(3)

    assert_array_equal(h.view(), [1,2,1])
    assert h.sum() == 4
