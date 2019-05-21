import pytest

import boost.histogram as bh
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

methods = [
    bh.hist.regular_int,
    bh.hist.regular_unlimited,
    bh.hist.any_int,
]

@pytest.mark.parametrize("hist_func", methods)
def test_1D_fill_int(hist_func):
    bins = 10
    ranges = (0, 1)

    vals = (.15, .25, .25)

    hist = hist_func([
        bh.axis.regular_uoflow(bins, *ranges)
        ])
    hist.fill(vals)

    H =  np.array([0, 1, 2, 0, 0, 0, 0, 0, 0, 0])

    assert_array_equal(np.asarray(hist), H)
    assert_array_equal(hist.view(flow=False), H)
    assert_array_equal(hist.view(flow=True)[1:-1], H)

    assert hist.axis(0).size() == bins
    assert hist.axis(0).size(flow=True) == bins + 2

@pytest.mark.parametrize("hist_func", methods)
def test_2D_fill_int(hist_func):
    bins = (10, 15)
    ranges = ((0, 3), (0, 2))

    vals = ((.15, .25, .25), (.35, .45, .45))

    hist = hist_func([
        bh.axis.regular_uoflow(bins[0], *ranges[0]),
        bh.axis.regular_uoflow(bins[1], *ranges[1]),
        ])
    hist.fill(*vals)

    H = np.histogram2d(*vals, bins=bins, range=ranges)[0]

    assert_array_equal(np.asarray(hist), H)
    assert_array_equal(hist.view(flow=True)[1:-1, 1:-1], H)
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
    hist.fill(vals)

    bins = np.asarray(hist)
    assert_array_equal(bins, [0,2,2])
    assert_array_equal(hist.view(flow=True), [0,0,2,2,0])
    assert_array_equal(hist.view(flow=False), [0,2,2])

def test_int_histogram():
    hist = bh.hist.any_int([
        bh.axis.integer_uoflow(3,7)
        ])

    vals = (1,2,3,4,5,6,7,8,9)
    hist.fill(vals)

    bins = np.asarray(hist)
    assert_array_equal(bins, [1,1,1,1])
    assert_array_equal(hist.view(flow=False), [1,1,1,1])
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

    hist.fill(1.45)

    assert hist.size() == 15

def test_numpy_flow():
    h = bh.hist.regular_int([bh.axis.regular_uoflow(10,0,1), bh.axis.regular_uoflow(5,0,1)])

    for i in range(10):
        for j in range(5):
            x,y = h.axis(0).bin(i).center(), h.axis(1).bin(j).center()
            v = i + j*10 + 1;
            h.fill([x]*v,[y]*v)

    flow_true = h.to_numpy(True)[0][1:-1, 1:-1]
    flow_false = h.to_numpy(False)[0]

    assert_array_equal(flow_true, flow_false)

    view_flow_true = h.view(flow=True)
    view_flow_false = h.view(flow=False)
    view_flow_default = h.view()

    assert_array_equal(view_flow_true[1:-1, 1:-1], view_flow_false)
    assert_array_equal(view_flow_default, view_flow_false)



def test_numpy_compare():
    h = bh.hist.regular_int([bh.axis.regular_uoflow(10,0,1), bh.axis.regular_uoflow(5,0,1)])

    xs = []
    ys = []
    for i in range(10):
        for j in range(5):
            x,y = h.axis(0).bin(i).center(), h.axis(1).bin(j).center()
            v = i + j*10 + 1;
            xs += [x]*v
            ys += [y]*v

    h.fill(xs, ys)

    H, E1, E2 = h.to_numpy()

    nH, nE1, nE2 = np.histogram2d(xs, ys, bins=(10,5), range=((0,1),(0,1)))

    assert_array_equal(H, nH)
    assert_allclose(E1, nE1)
    assert_allclose(E2, nE2)

def test_project():
    h = bh.hist.regular_int([bh.axis.regular_uoflow(10,0,1), bh.axis.regular_uoflow(5,0,1)])
    h0 = bh.hist.regular_int([bh.axis.regular_uoflow(10,0,1)])
    h1 = bh.hist.regular_int([bh.axis.regular_uoflow(5,0,1)])

    for x,y in ((.3,.3),(.7,.7),(.5,.6),(.23,.92),(.15,.32),(.43,.54)):
        h.fill(x,y)
        h0.fill(x)
        h1.fill(y)

    assert h.project(0, 1) == h
    assert h.project(0) == h0
    assert h.project(1) == h1

    assert_array_equal(h.project(0, 1), h)
    assert_array_equal(h.project(0), h0)
    assert_array_equal(h.project(1), h1)

def test_sums():
    h = bh.histogram(bh.axis.regular_uoflow(4,0,1))
    h.fill([.1,.2,.3,10])

    assert h.sum() == 3
    assert h.sum(flow=True) == 4

def test_int_cat_hist():
    h = bh.hist.any_int([bh.axis.category_int([1,2,3])])

    h.fill(1)
    h.fill(2)
    h.fill(2.2)
    h.fill(3)

    assert_array_equal(h.view(), [1,2,1])
    assert h.sum() == 4
