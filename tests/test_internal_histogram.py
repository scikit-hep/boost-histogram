import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx

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


@pytest.mark.filterwarnings("ignore:List indexing selection is experimental")
def test_int_cat_hist_pick_several():
    h = bh.Histogram(
        bh.axis.IntCategory([1, 2, 7], __dict__={"xval": 5}), storage=bh.storage.Int64()
    )

    h.fill(1, weight=8)
    h.fill(2, weight=7)
    h.fill(7, weight=6)

    assert h.view() == approx(np.array([8, 7, 6]))
    assert h.sum() == 21

    assert h[[0, 2]].values() == approx(np.array([8, 6]))
    assert h[[2, 0]].values() == approx(np.array([6, 8]))
    assert h[[1]].values() == approx(np.array([7]))

    assert h[[bh.loc(1), bh.loc(7)]].values() == approx(np.array([8, 6]))
    assert h[[bh.loc(7), bh.loc(1)]].values() == approx(np.array([6, 8]))
    assert h[[bh.loc(2)]].values() == approx(np.array([7]))

    assert tuple(h[[0, 2]].axes[0]) == (1, 7)
    assert tuple(h[[2, 0]].axes[0]) == (7, 1)
    assert tuple(h[[1]].axes[0]) == (2,)

    assert h.axes[0].xval == 5
    assert h[[0]].axes[0].xval == 5
    assert h[[0, 1, 2]].axes[0].xval == 5


@pytest.mark.filterwarnings("ignore:List indexing selection is experimental")
def test_str_cat_pick_several():
    h = bh.Histogram(bh.axis.StrCategory(["a", "b", "c"]))

    h.fill(["a", "a", "a", "b", "b", "c"], weight=0.25)

    assert h[[0, 1, 2]].values() == approx(np.array([0.75, 0.5, 0.25]))
    assert h[[2, 1, 0]].values() == approx(np.array([0.25, 0.5, 0.75]))
    assert h[[1, 0]].values() == approx(np.array([0.5, 0.75]))

    assert h[[bh.loc("a"), bh.loc("b"), bh.loc("c")]].values() == approx(
        np.array([0.75, 0.5, 0.25])
    )
    assert h[[bh.loc("c"), bh.loc("b"), bh.loc("a")]].values() == approx(
        np.array([0.25, 0.5, 0.75])
    )
    assert h[[bh.loc("b"), bh.loc("a")]].values() == approx(np.array([0.5, 0.75]))

    assert tuple(h[[1, 0]].axes[0]) == ("b", "a")


@pytest.mark.filterwarnings("ignore:List indexing selection is experimental")
def test_pick_invalid():
    h = bh.Histogram(bh.axis.Regular(10, 0, 1))
    with pytest.raises(RuntimeError):
        h[[0, 1]]

    h = bh.Histogram(bh.axis.Integer(0, 10))
    with pytest.raises(RuntimeError):
        h[[0, 1]]


@pytest.mark.filterwarnings("ignore:List indexing selection is experimental")
def test_str_cat_pick_dual():
    h = bh.Histogram(
        bh.axis.StrCategory(["a", "b", "c"]), bh.axis.StrCategory(["d", "e", "f"])
    )
    vals = np.arange(9).reshape(3, 3)
    h.values()[...] = vals

    assert h[[0], [0]].values() == approx(vals[[0]][:, [0]])
    assert h[[1], [2]].values() == approx(vals[[1]][:, [2]])
    assert h[[1], [0, 1]].values() == approx(vals[[1]][:, [0, 1]])
    assert h[[0, 1], [1]].values() == approx(vals[[0, 1]][:, [1]])
    assert h[[0, 1], [0, 1]].values() == approx(vals[[0, 1]][:, [0, 1]])
    assert h[[0, 1], [2, 1]].values() == approx(vals[[0, 1]][:, [2, 1]])


@pytest.mark.filterwarnings("ignore:List indexing selection is experimental")
def test_pick_multiaxis():
    h = bh.Histogram(
        bh.axis.StrCategory(["a", "b", "c"]),
        bh.axis.IntCategory([-5, 0, 10]),
        bh.axis.Regular(5, 0, 1),
        bh.axis.StrCategory(["d", "e", "f"]),
        storage=bh.storage.Int64(),
    )

    h.fill("b", -5, 0.65, "f")
    h.fill("b", -5, 0.65, "e")

    mini = h[[bh.loc("b"), 2], [1, bh.loc(-5)], sum, bh.loc("f")]

    assert mini.ndim == 2
    assert tuple(mini.axes[0]) == ("b", "c")
    assert tuple(mini.axes[1]) == (0, -5)

    assert h[[1, 2], [0, 1], sum, bh.loc("f")].sum() == 1
    assert h[[1, 2], [1, 0], sum, bh.loc("f")].sum() == 1

    assert mini.values() == approx(np.array(((0, 1), (0, 0))))
