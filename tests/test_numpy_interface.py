# -*- coding: utf-8 -*-
import copy

import numpy as np
import pytest

import boost_histogram as bh

np113 = tuple(int(x) for x in np.__version__.split(".")[:2]) >= (1, 13)

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
    {"bins": "auto" if np113 else 20},
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

    np.testing.assert_array_almost_equal(e1, e2)
    np.testing.assert_array_equal(h1, h2)

    opt = copy.deepcopy(opt)
    opt["density"] = True

    h1, e1 = np.histogram(v, **opt)
    h2, e2 = bh.numpy.histogram(v, **opt)

    np.testing.assert_array_almost_equal(e1, e2)
    np.testing.assert_array_almost_equal(h1, h2)


@pytest.mark.parametrize("a", inputs_1d)
@pytest.mark.parametrize("opt", opts)
def test_histogram1d_object(a, opt):
    bh_opt = copy.deepcopy(opt)
    bh_opt["histogram"] = bh.Histogram

    v = np.array(a)

    h1, e1 = np.histogram(v, **opt)
    bh_h2 = bh.numpy.histogram(v, **bh_opt)
    h2, e2 = bh_h2.to_numpy()

    np.testing.assert_array_almost_equal(e1, e2)
    np.testing.assert_array_equal(h1, h2)

    # Ensure reducible
    assert bh_h2[:5].values() == pytest.approx(h1[:5])

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

    np.testing.assert_array_almost_equal(e1x, e2x)
    np.testing.assert_array_almost_equal(e1y, e2y)
    np.testing.assert_array_equal(h1, h2)

    h1, e1x, e1y = np.histogram2d(x, y, density=True)
    h2, e2x, e2y = bh.numpy.histogram2d(x, y, density=True)

    np.testing.assert_array_almost_equal(e1x, e2x)
    np.testing.assert_array_almost_equal(e1y, e2y)
    np.testing.assert_array_almost_equal(h1, h2)


def test_histogram2d_object():
    x = np.array([0.3, 0.3, 0.1, 0.8, 0.34, 0.03, 0.32, 0.65])
    y = np.array([0.4, 0.5, 0.22, 0.65, 0.32, 0.01, 0.23, 1.98])

    h1, e1x, e1y = np.histogram2d(x, y)
    bh_h2 = bh.numpy.histogram2d(x, y, histogram=bh.Histogram)
    h2, e2x, e2y = bh_h2.to_numpy()

    np.testing.assert_array_almost_equal(e1x, e2x)
    np.testing.assert_array_almost_equal(e1y, e2y)
    np.testing.assert_array_equal(h1, h2)

    with pytest.raises(KeyError):
        bh.numpy.histogram2d(x, y, density=True, histogram=bh.Histogram)


def test_histogramdd():
    x = np.array([0.3, 0.3, 0.1, 0.8, 0.34, 0.03, 0.32, 0.65])
    y = np.array([0.4, 0.5, 0.22, 0.65, 0.32, 0.01, 0.23, 1.98])
    z = np.array([0.5, 0.7, 0.0, 0.65, 0.72, 0.01, 0.3, 1.4])

    h1, (e1x, e1y, e1z) = np.histogramdd([x, y, z])
    h2, (e2x, e2y, e2z) = bh.numpy.histogramdd([x, y, z])

    np.testing.assert_array_almost_equal(e1x, e2x)
    np.testing.assert_array_almost_equal(e1y, e2y)
    np.testing.assert_array_almost_equal(e1z, e2z)
    np.testing.assert_array_equal(h1, h2)

    h1, (e1x, e1y, e1z) = np.histogramdd([x, y, z], density=True)
    h2, (e2x, e2y, e2z) = bh.numpy.histogramdd([x, y, z], density=True)

    np.testing.assert_array_almost_equal(e1x, e2x)
    np.testing.assert_array_almost_equal(e1y, e2y)
    np.testing.assert_array_almost_equal(e1z, e2z)
    np.testing.assert_array_almost_equal(h1, h2)


def test_histogramdd_object():
    x = np.array([0.3, 0.3, 0.1, 0.8, 0.34, 0.03, 0.32, 0.65])
    y = np.array([0.4, 0.5, 0.22, 0.65, 0.32, 0.01, 0.23, 1.98])
    z = np.array([0.5, 0.7, 0.0, 0.65, 0.72, 0.01, 0.3, 1.4])

    h1, (e1x, e1y, e1z) = np.histogramdd([x, y, z])
    bh_h2 = bh.numpy.histogramdd([x, y, z], histogram=bh.Histogram)
    h2, (e2x, e2y, e2z) = bh_h2.to_numpy(dd=True)

    np.testing.assert_array_almost_equal(e1x, e2x)
    np.testing.assert_array_almost_equal(e1y, e2y)
    np.testing.assert_array_almost_equal(e1z, e2z)
    np.testing.assert_array_equal(h1, h2)

    with pytest.raises(KeyError):
        bh.numpy.histogramdd([x, y, z], density=True, histogram=bh.Histogram)


def test_histogram_weights():
    x = np.array([0.3, 0.3, 0.1, 0.8, 0.34, 0.03, 0.32, 0.65])
    weights = np.array([0.4, 0.5, 0.22, 0.65, 0.32, 0.01, 0.23, 1.98])
    h1, edges = np.histogram(x, weights=weights)
    bh_h1, edges = bh.numpy.histogram(x, weights=weights)

    np.testing.assert_array_almost_equal(h1, bh_h1)
