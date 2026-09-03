from __future__ import annotations

import copy

import numpy as np
import pytest
from pytest import approx

import boost_histogram as bh

inputs_1d = (
    [1, 2, 3, 4, 3, 4, 5, 10, 9, 11, 21, -2],
    [
        0.27237556,
        0.72020987,
        0.75204098,
        -0.29265003,
        -2.67332888,
        0.68420365,
        -0.60629843,
        -0.6375687,
        -1.00017927,
        -0.07707552,
    ],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
)
opts = (
    {},
    {"bins": 10},
    {"bins": "auto"},
    {"range": (0, 5), "bins": 30},
    {"range": np.array((0, 5), dtype=float), "bins": np.int32(30)},
    {"range": np.array((0, 3), dtype=np.double), "bins": np.uint32(10)},
    {"range": np.array((0, 10), dtype=int), "bins": np.int8(30)},
    {"bins": [0, 1, 1.2, 1.3, 4, 21]},
)


@pytest.mark.parametrize("a", inputs_1d)
@pytest.mark.parametrize("opt", opts)
def test_histogram1d(a, opt):
    v = np.array(a)

    h1, e1 = np.histogram(v, **opt)
    h2, e2 = bh.numpy.histogram(v, **opt)

    assert e1 == approx(e2)
    assert h1 == approx(h2)

    opt = copy.deepcopy(opt)
    opt["density"] = True

    h1, e1 = np.histogram(v, **opt)
    h2, e2 = bh.numpy.histogram(v, **opt)

    assert e1 == approx(e2)
    assert h1 == approx(h2)


@pytest.mark.parametrize("a", inputs_1d)
@pytest.mark.parametrize("opt", opts)
def test_histogram1d_object(a, opt):
    bh_opt = copy.deepcopy(opt)
    bh_opt["histogram"] = bh.Histogram

    v = np.array(a)

    h1, e1 = np.histogram(v, **opt)
    bh_h2 = bh.numpy.histogram(v, **bh_opt)
    h2, e2 = bh_h2.to_numpy()

    assert e1 == approx(e2)
    assert h1 == approx(h2)

    # Ensure reducible
    assert bh_h2[:5].values() == approx(h1[:5])

    opt = copy.deepcopy(opt)
    opt["density"] = True

    bh_opt = copy.deepcopy(bh_opt)
    bh_opt["density"] = True

    with pytest.raises(KeyError):
        bh_h2 = bh.numpy.histogram(v, **bh_opt)


def test_histogram2d():
    x = np.array([0.3, 0.3, 0.1, 0.8, 0.34, 0.03, 0.32, 0.65])
    y = np.array([0.4, 0.5, 0.22, 0.65, 0.32, 0.01, 0.23, 1.98])

    h1, e1x, e1y = np.histogram2d(x, y)
    h2, e2x, e2y = bh.numpy.histogram2d(x, y)

    assert e1x == approx(e2x)
    assert e1y == approx(e2y)
    assert h1 == approx(h2)

    h1, e1x, e1y = np.histogram2d(x, y, density=True)
    h2, e2x, e2y = bh.numpy.histogram2d(x, y, density=True)

    assert e1x == approx(e2x)
    assert e1y == approx(e2y)
    assert h1 == approx(h2)


def test_histogram2d_object():
    x = np.array([0.3, 0.3, 0.1, 0.8, 0.34, 0.03, 0.32, 0.65])
    y = np.array([0.4, 0.5, 0.22, 0.65, 0.32, 0.01, 0.23, 1.98])

    h1, e1x, e1y = np.histogram2d(x, y)
    bh_h2 = bh.numpy.histogram2d(x, y, histogram=bh.Histogram)
    h2, e2x, e2y = bh_h2.to_numpy()

    assert e1x == approx(e2x)
    assert e1y == approx(e2y)
    assert h1 == approx(h2)

    with pytest.raises(KeyError):
        bh.numpy.histogram2d(x, y, density=True, histogram=bh.Histogram)


def test_histogramdd():
    x = np.array([0.3, 0.3, 0.1, 0.8, 0.34, 0.03, 0.32, 0.65])
    y = np.array([0.4, 0.5, 0.22, 0.65, 0.32, 0.01, 0.23, 1.98])
    z = np.array([0.5, 0.7, 0.0, 0.65, 0.72, 0.01, 0.3, 1.4])

    h1, (e1x, e1y, e1z) = np.histogramdd([x, y, z])
    h2, (e2x, e2y, e2z) = bh.numpy.histogramdd([x, y, z])

    assert e1x == approx(e2x)
    assert e1y == approx(e2y)
    assert e1z == approx(e2z)
    assert h1 == approx(h2)

    h1, (e1x, e1y, e1z) = np.histogramdd([x, y, z], density=True)
    h2, (e2x, e2y, e2z) = bh.numpy.histogramdd([x, y, z], density=True)

    assert e1x == approx(e2x)
    assert e1y == approx(e2y)
    assert e1z == approx(e2z)
    assert h1 == approx(h2)


def test_histogramdd_object():
    x = np.array([0.3, 0.3, 0.1, 0.8, 0.34, 0.03, 0.32, 0.65])
    y = np.array([0.4, 0.5, 0.22, 0.65, 0.32, 0.01, 0.23, 1.98])
    z = np.array([0.5, 0.7, 0.0, 0.65, 0.72, 0.01, 0.3, 1.4])

    h1, (e1x, e1y, e1z) = np.histogramdd([x, y, z])
    bh_h2 = bh.numpy.histogramdd([x, y, z], histogram=bh.Histogram)
    h2, (e2x, e2y, e2z) = bh_h2.to_numpy(dd=True)

    assert e1x == approx(e2x)
    assert e1y == approx(e2y)
    assert e1z == approx(e2z)
    assert h1 == approx(h2)

    with pytest.raises(KeyError):
        bh.numpy.histogramdd([x, y, z], density=True, histogram=bh.Histogram)


def test_histogram_weights():
    x = np.array([0.3, 0.3, 0.1, 0.8, 0.34, 0.03, 0.32, 0.65])
    weights = np.array([0.4, 0.5, 0.22, 0.65, 0.32, 0.01, 0.23, 1.98])
    h1, edges = np.histogram(x, weights=weights)
    bh_h1, bh_edges = bh.numpy.histogram(x, weights=weights)

    assert bh_h1 == approx(h1)
    assert bh_edges == approx(edges)


def test_histogram_nans():
    x = np.array([0, 1, 2, 3, np.nan])

    with pytest.raises(ValueError):
        np.histogram(x)

    with pytest.raises(ValueError):
        bh.numpy.histogram(x)


def test_histogram_all_zeros():
    x = np.array([0, 0, 0, 0, 0, 0])
    h1, edges = np.histogram(x)
    bh_h1, bh_edges = bh.numpy.histogram(x)

    assert bh_h1 == approx(h1)
    assert bh_edges == approx(edges)


def test_histogram_all_ones():
    x = np.array([1, 1, 1, 1, 1, 1])
    h1, edges = np.histogram(x)
    bh_h1, bh_edges = bh.numpy.histogram(x)

    assert bh_h1 == approx(h1)
    assert bh_edges == approx(edges)


def test_histogram_does_not_mutate_input_edges():
    edges = np.array([0.0, 1.0, 2.0])
    original = edges.copy()

    bh.numpy.histogram([0.5, 1.5], bins=edges)

    assert edges == approx(original)


def test_histogramdd_does_not_mutate_input_edges():
    x = np.array([0.3, 0.3, 0.1, 0.8])
    y = np.array([0.4, 0.5, 0.22, 0.65])
    edges_x = np.array([0.0, 0.5, 1.0])
    edges_y = np.array([0.0, 0.25, 0.5, 1.0])
    original_x = edges_x.copy()
    original_y = edges_y.copy()

    bh.numpy.histogramdd((x, y), bins=(edges_x, edges_y))

    assert edges_x == approx(original_x)
    assert edges_y == approx(original_y)


@pytest.mark.parametrize("bad_bins", [(2,), (2, 2, 2), (2, 2, 2, 2)])
def test_histogramdd_mismatched_bins(bad_bins):
    x = np.array([0.3, 0.3, 0.1, 0.8])
    y = np.array([0.4, 0.5, 0.22, 0.65])

    with pytest.raises(ValueError, match="dimension of bins"):
        np.histogramdd((x, y), bins=bad_bins)

    with pytest.raises(ValueError, match="dimension of bins"):
        bh.numpy.histogramdd((x, y), bins=bad_bins)


@pytest.mark.parametrize(
    "bad_range", [((0, 1),), ((0, 1), (0, 1), (0, 1)), ((0, 1), (0, 1), (0, 1), (0, 1))]
)
def test_histogramdd_mismatched_range(bad_range):
    x = np.array([0.3, 0.3, 0.1, 0.8])
    y = np.array([0.4, 0.5, 0.22, 0.65])

    with pytest.raises(ValueError, match="range argument"):
        np.histogramdd((x, y), bins=2, range=bad_range)

    with pytest.raises(ValueError, match="range argument"):
        bh.numpy.histogramdd((x, y), bins=2, range=bad_range)


@pytest.mark.parametrize("empty", [[], np.array([]), np.array([], dtype=np.int64)])
@pytest.mark.parametrize("opt", [{}, {"bins": 5}, {"range": (0, 4)}])
def test_histogram_empty(empty, opt):
    h1, e1 = np.histogram(empty, **opt)
    h2, e2 = bh.numpy.histogram(empty, **opt)

    assert e1 == approx(e2)
    assert h1 == approx(h2)


def test_histogramdd_empty():
    h1, (e1,) = np.histogramdd((np.array([]),), bins=(4,))
    h2, (e2,) = bh.numpy.histogramdd((np.array([]),), bins=(4,))

    assert e1 == approx(e2)
    assert h1 == approx(h2)


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_histogramdd_1d_array(dtype):
    v = np.array([1, 2, 3, 4, 3, 4, 5, 10, 9, 11, 21, -2], dtype=dtype)

    h1, (e1,) = np.histogramdd(v)
    h2, (e2,) = bh.numpy.histogramdd(v)

    assert e1 == approx(e2)
    assert h1 == approx(h2)


def test_histogramdd_1d_list():
    v = [1.0, 2.0, 3.0, 4.0]

    h1, (e1,) = np.histogramdd(v, bins=3)
    h2, (e2,) = bh.numpy.histogramdd(v, bins=3)

    assert e1 == approx(e2)
    assert h1 == approx(h2)


@pytest.mark.parametrize("bins", [np.array([0.0, 0.5, 1.0, 2.0]), [0.0, 0.5, 1.0, 2.0]])
def test_histogram2d_shared_edges(bins):
    x = np.array([0.3, 0.3, 0.1, 0.8, 0.34, 0.03, 0.32, 0.65])
    y = np.array([0.4, 0.5, 0.22, 0.65, 0.32, 0.01, 0.23, 1.98])

    h1, e1x, e1y = np.histogram2d(x, y, bins=bins)
    h2, e2x, e2y = bh.numpy.histogram2d(x, y, bins=bins)

    assert e1x == approx(e2x)
    assert e1y == approx(e2y)
    assert h1 == approx(h2)


def test_histogram_reversed_range():
    x = np.array([0.3, 0.3, 0.1, 0.8])

    with pytest.raises(ValueError, match="max must be larger than min"):
        np.histogram(x, bins=4, range=(1, 0))

    with pytest.raises(ValueError, match="max must be larger than min"):
        bh.numpy.histogram(x, bins=4, range=(1, 0))


def test_histogramdd_reversed_range():
    x = np.array([0.3, 0.3, 0.1, 0.8])
    y = np.array([0.4, 0.5, 0.22, 0.65])

    with pytest.raises(ValueError, match="max must be larger than min"):
        np.histogramdd((x, y), bins=4, range=((0, 1), (1, 0)))

    with pytest.raises(ValueError, match="max must be larger than min"):
        bh.numpy.histogramdd((x, y), bins=4, range=((0, 1), (1, 0)))


def test_histogram_nonfinite_range():
    x = np.array([0.3, 0.3, 0.1, 0.8])

    with pytest.raises(ValueError, match="is not finite"):
        np.histogram(x, bins=4, range=(0, np.inf))

    with pytest.raises(ValueError, match="is not finite"):
        bh.numpy.histogram(x, bins=4, range=(0, np.inf))


def test_histogram_empty_range():
    x = np.array([0.3, 0.3, 0.1, 0.8])

    h1, e1 = np.histogram(x, bins=4, range=(1, 1))
    h2, e2 = bh.numpy.histogram(x, bins=4, range=(1, 1))

    assert e1 == approx(e2)
    assert h1 == approx(h2)


@pytest.mark.parametrize("opt", [{}, {"density": True}])
def test_histogram_weights_and_density(opt):
    x = np.array([0.3, 0.3, 0.1, 0.8, 0.34, 0.03, 0.32, 0.65])
    weights = np.array([0.4, 0.5, 0.22, 0.65, 0.32, 0.01, 0.23, 1.98])

    h1, e1 = np.histogram(x, bins=7, weights=weights, **opt)
    h2, e2 = bh.numpy.histogram(x, bins=7, weights=weights, **opt)

    assert e1 == approx(e2)
    assert h1 == approx(h2)

    h1, e1x, e1y = np.histogram2d(x, x, bins=3, weights=weights, **opt)
    h2, e2x, e2y = bh.numpy.histogram2d(x, x, bins=3, weights=weights, **opt)

    assert e1x == approx(e2x)
    assert e1y == approx(e2y)
    assert h1 == approx(h2)


@pytest.mark.xfail(
    reason="A Regular axis finds the bin from the range, so a value exactly on "
    "an interior edge can land one bin lower than in NumPy, which compares the "
    "value against the linspace edges."
)
@pytest.mark.parametrize(("lo", "hi", "n"), [(-3, 7, 13), (0, 1, 7), (0, 3, 11)])
def test_histogram_values_on_edges(lo, hi, n):
    v = np.linspace(lo, hi, n + 1)

    h1, e1 = np.histogram(v, bins=n, range=(lo, hi))
    h2, e2 = bh.numpy.histogram(v, bins=n, range=(lo, hi))

    assert e1.tolist() == e2.tolist()
    assert h1.tolist() == h2.tolist()
