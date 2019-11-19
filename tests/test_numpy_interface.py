import pytest

import boost_histogram as bh
import numpy as np

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import boost_histogram.numpy as bhnp

np113 = tuple(int(x) for x in np.__version__.split(".")[:2]) >= (1, 13)


@pytest.mark.parametrize(
    "a",
    (
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
    ),
)
@pytest.mark.parametrize(
    "opt",
    (
        {},
        {"bins": 10},
        {"bins": "auto" if np113 else 20},
        {"range": (0, 5), "bins": 30},
        {"bins": [0, 1, 1.2, 1.3, 4, 21]},
    ),
)
def test_histogram1d(a, opt):
    v = np.array(a)
    h1, e1 = np.histogram(v, **opt)
    h2, e2 = bhnp.histogram(v, **opt)

    np.testing.assert_array_almost_equal(e1, e2)
    np.testing.assert_array_equal(h1, h2)


def test_histogram2d():
    x = np.array([0.3, 0.3, 0.1, 0.8, 0.34, 0.03, 0.32, 0.65])
    y = np.array([0.4, 0.5, 0.22, 0.65, 0.32, 0.01, 0.23, 1.98])

    h1, e1x, e1y = np.histogram2d(x, y)
    h2, e2x, e2y = bhnp.histogram2d(x, y)

    np.testing.assert_array_almost_equal(e1x, e2x)
    np.testing.assert_array_almost_equal(e1y, e2y)
    np.testing.assert_array_equal(h1, h2)
