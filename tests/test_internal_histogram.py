# -*- coding: utf-8 -*-
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import boost_histogram as bh

methods = (bh.storage.Double, bh.storage.Int64, bh.storage.Unlimited)


@pytest.mark.parametrize("dtype", [np.double, np.int_, np.float32])
def test_noncontig_fill(dtype):
    a = np.array([[0, 0], [1, 1]], dtype=dtype, order="C")
    b = np.array([[0, 0], [1, 1]], dtype=dtype, order="F")

    h1 = bh.Histogram(bh.axis.Regular(10, 0, 2)).fill(a[0])
    h2 = bh.Histogram(bh.axis.Regular(10, 0, 2)).fill(b[0])

    assert h1 == h2


@pytest.mark.parametrize("storage", methods)
def test_1D_fill_int(storage):
    bins = 10
    ranges = (0, 1)

    vals = (0.15, 0.25, 0.25)

    hist = bh.Histogram(bh.axis.Regular(bins, *ranges), storage=storage())
    assert hist._storage_type == storage
    hist.fill(vals)

    H = np.array([0, 1, 2, 0, 0, 0, 0, 0, 0, 0])

    assert_array_equal(np.asarray(hist), H)
    assert_array_equal(hist.view(flow=False), H)
    assert_array_equal(hist.view(flow=True)[1:-1], H)

    assert hist.axes[0].size == bins
    assert hist.axes[0].extent == bins + 2


@pytest.mark.parametrize("storage", methods)
def test_2D_fill_int(storage):
    bins = (10, 15)
    ranges = ((0, 3), (0, 2))

    vals = ((0.15, 0.25, 0.25), (0.35, 0.45, 0.45))

    hist = bh.Histogram(
        bh.axis.Regular(bins[0], *ranges[0]),
        bh.axis.Regular(bins[1], *ranges[1]),
        storage=storage(),
    )
    assert hist._storage_type == storage
    hist.fill(*vals)

    H = np.histogram2d(*vals, bins=bins, range=ranges)[0]

    assert_array_equal(np.asarray(hist), H)
    assert_array_equal(hist.view(flow=True)[1:-1, 1:-1], H)
    assert_array_equal(hist.view(flow=False), H)

    assert hist.axes[0].size == bins[0]
    assert hist.axes[0].extent == bins[0] + 2

    assert hist.axes[1].size == bins[1]
    assert hist.axes[1].extent == bins[1] + 2


def test_edges_histogram():
    edges = (1, 12, 22, 79)
    hist = bh.Histogram(bh.axis.Variable(edges), storage=bh.storage.Int64())

    vals = (13, 15, 24, 29)
    hist.fill(vals)

    bins = np.asarray(hist)
    assert_array_equal(bins, [0, 2, 2])
    assert_array_equal(hist.view(flow=True), [0, 0, 2, 2, 0])
    assert_array_equal(hist.view(flow=False), [0, 2, 2])


def test_int_histogram():
    hist = bh.Histogram(bh.axis.Integer(3, 7), storage=bh.storage.Int64())

    vals = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    hist.fill(vals)

    bins = np.asarray(hist)
    assert_array_equal(bins, [1, 1, 1, 1])
    assert_array_equal(hist.view(flow=False), [1, 1, 1, 1])
    assert_array_equal(hist.view(flow=True), [2, 1, 1, 1, 1, 3])


def test_str_categories_histogram():
    hist = bh.Histogram(
        bh.axis.StrCategory(["a", "b", "c"]), storage=bh.storage.Int64()
    )

    vals = ["a", "b", "b", "c"]

    hist.fill(vals)

    assert hist[bh.loc("a")] == 1
    assert hist[bh.loc("b")] == 2
    assert hist[bh.loc("c")] == 1


def test_growing_histogram():
    hist = bh.Histogram(
        bh.axis.Regular(10, 0, 1, growth=True), storage=bh.storage.Int64()
    )

    hist.fill(1.45)

    assert hist.size == 17


def test_numpy_dd():
    h = bh.Histogram(
        bh.axis.Regular(10, 0, 1), bh.axis.Regular(5, 0, 1), storage=bh.storage.Int64()
    )

    for i in range(10):
        for j in range(5):
            x, y = h.axes[0].centers[i], h.axes[1].centers[j]
            v = i + j * 10 + 1
            h.fill([x] * v, [y] * v)

    h2, x2, y2 = h.to_numpy()
    h1, (x1, y1) = h.to_numpy(dd=True)

    assert_array_equal(h1, h2)
    assert_array_equal(x1, x2)
    assert_array_equal(y1, y2)

def test_numpy_weights():
    h = bh.Histogram(
        bh.axis.Regular(10, 0, 1), bh.axis.Regular(5, 0, 1), storage=bh.storage.Weight()
    )

    for i in range(10):
        for j in range(5):
            x, y = h.axes[0].centers[i], h.axes[1].centers[j]
            v = i + j * 10 + 1
            h.fill([x] * v, [y] * v)

    h2, x2, y2 = h.to_numpy(view=False)
    h1, (x1, y1) = h.to_numpy(dd=True, view=False)

    assert_array_equal(h1, h2)
    assert_array_equal(x1, x2)
    assert_array_equal(y1, y2)

    h1, (x1, y1) = h.to_numpy(dd=True, view=False)
    h2, x2, y2 = h.to_numpy(view=True)

    assert_array_equal(h1, h2.value)
    assert_array_equal(x1, x2)
    assert_array_equal(y1, y2)


def test_numpy_flow():
    h = bh.Histogram(
        bh.axis.Regular(10, 0, 1), bh.axis.Regular(5, 0, 1), storage=bh.storage.Int64()
    )

    for i in range(10):
        for j in range(5):
            x, y = h.axes[0].centers[i], h.axes[1].centers[j]
            v = i + j * 10 + 1
            h.fill([x] * v, [y] * v)

    flow_true = h.to_numpy(True)[0][1:-1, 1:-1]
    flow_false = h.to_numpy(False)[0]

    assert_array_equal(flow_true, flow_false)

    view_flow_true = h.view(flow=True)
    view_flow_false = h.view(flow=False)
    view_flow_default = h.view()

    assert_array_equal(view_flow_true[1:-1, 1:-1], view_flow_false)
    assert_array_equal(view_flow_default, view_flow_false)


def test_numpy_compare():
    h = bh.Histogram(
        bh.axis.Regular(10, 0, 1), bh.axis.Regular(5, 0, 1), storage=bh.storage.Int64()
    )

    xs = []
    ys = []
    for i in range(10):
        for j in range(5):
            x, y = h.axes[0].centers[i], h.axes[1].centers[j]
            v = i + j * 10 + 1
            xs += [x] * v
            ys += [y] * v

    h.fill(xs, ys)

    H, E1, E2 = h.to_numpy()

    nH, nE1, nE2 = np.histogram2d(xs, ys, bins=(10, 5), range=((0, 1), (0, 1)))

    assert_array_equal(H, nH)
    assert_allclose(E1, nE1)
    assert_allclose(E2, nE2)


def test_project():
    h = bh.Histogram(
        bh.axis.Regular(10, 0, 1), bh.axis.Regular(5, 0, 1), storage=bh.storage.Int64()
    )
    h0 = bh.Histogram(bh.axis.Regular(10, 0, 1), storage=bh.storage.Int64())
    h1 = bh.Histogram(bh.axis.Regular(5, 0, 1), storage=bh.storage.Int64())

    for x, y in (
        (0.3, 0.3),
        (0.7, 0.7),
        (0.5, 0.6),
        (0.23, 0.92),
        (0.15, 0.32),
        (0.43, 0.54),
    ):
        h.fill(x, y)
        h0.fill(x)
        h1.fill(y)

    assert h.project(0, 1) == h
    assert h.project(0) == h0
    assert h.project(1) == h1

    assert_array_equal(h.project(0, 1), h)
    assert_array_equal(h.project(0), h0)
    assert_array_equal(h.project(1), h1)


def test_sums():
    h = bh.Histogram(bh.axis.Regular(4, 0, 1))
    h.fill([0.1, 0.2, 0.3, 10])

    assert h.sum() == 3
    assert h.sum(flow=True) == 4


def test_int_cat_hist():
    h = bh.Histogram(bh.axis.IntCategory([1, 2, 3]), storage=bh.storage.Int64())

    h.fill(1)
    h.fill(2)
    h.fill(3)

    assert_array_equal(h.view(), [1, 1, 1])
    assert h.sum() == 3

    with pytest.raises(RuntimeError):
        h.fill(0.5)
