from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

import boost_histogram as bh


def test_1D_set_bin():
    h = bh.Histogram(bh.axis.Regular(10, 0, 1))

    h[2] = 2
    assert h[2] == 2.0

    h[bh.underflow] = 3
    assert h[bh.underflow] == 3.0

    h[bh.overflow] = 4
    assert h[bh.overflow] == 4.0


def test_2d_set_bin():
    h = bh.Histogram(bh.axis.Regular(10, 0, 1), bh.axis.Regular(10, 0, 1))

    h[2, 4] = 2
    assert h[2, 4] == 2.0

    h[bh.underflow, 5] = 3
    assert h[bh.underflow, 5] == 3.0

    h[bh.overflow, bh.overflow] = 4
    assert h[bh.overflow, bh.overflow] == 4.0


def test_1d_set_array():
    h = bh.Histogram(bh.axis.Regular(10, 0, 1))

    h[...] = np.arange(10)
    assert h.view() == approx(np.arange(10))

    h[...] = np.arange(12)
    assert h.view(flow=True) == approx(np.arange(12))

    with pytest.raises(ValueError):
        h[...] = np.arange(9)
    with pytest.raises(ValueError):
        h[...] = np.arange(11)
    with pytest.raises(ValueError):
        h[...] = np.arange(13)

    h[...] = 1
    assert h.view() == approx(np.ones(10))


def test_2d_set_array():
    h = bh.Histogram(bh.axis.Regular(10, 0, 1), bh.axis.Regular(10, 0, 1))

    h[...] = np.arange(10).reshape(-1, 1)
    assert h.view()[:, 2] == approx(np.arange(10))

    h[...] = np.arange(12).reshape(-1, 1)
    assert h.view(flow=True)[:, 3] == approx(np.arange(12))

    with pytest.raises(ValueError):
        h[...] = np.arange(9).reshape(-1, 1)
    with pytest.raises(ValueError):
        h[...] = np.arange(11).reshape(-1, 1)
    with pytest.raises(ValueError):
        h[...] = np.arange(13).reshape(-1, 1)

    h[...] = 1
    assert h.view() == approx(np.ones((10, 10)))


def test_weighted_set_shortcut():
    h = bh.Histogram(bh.axis.Regular(10, 0, 1), storage=bh.storage.Weight())

    h[0] = (1, 2)

    assert h[0].value == 1
    assert h[0].variance == 2

    h[3:6] = ((3, 4), (5, 6), (7, 8))

    assert h[3].value == 3
    assert h[3].variance == 4
    assert h[4].value == 5
    assert h[4].variance == 6
    assert h[5].value == 7
    assert h[5].variance == 8


@pytest.mark.parametrize(
    ("storage", "default"),
    [
        (bh.storage.Mean, bh.accumulators.Mean(1.0, 2.0, 3.0)),
        (bh.storage.WeightedMean, bh.accumulators.WeightedMean(1.0, 2.0, 3.0, 4.0)),
        (bh.storage.Weight, bh.accumulators.WeightedSum(1.0, 2)),
    ],
)
def test_set_special_dtype(storage, default):
    h = bh.Histogram(
        bh.axis.Regular(10, 0, 1), bh.axis.Regular(10, 0, 1), storage=storage()
    )

    arr = np.full((10, 1), default)
    h[...] = arr
    assert h.view()[:, 1:2] == approx(arr)

    arr = np.full((12, 1), default)
    h[...] = arr
    assert h.view(flow=True)[:, 2:3] == approx(arr)

    arr = np.full((10, 10), default)
    h[...] = arr
    assert h.view() == approx(arr)

    arr = np.full((10, 12), default)
    h[...] = arr
    assert h.view(flow=True)[1:11, :] == approx(arr)

    arr = np.full((12, 10), default)
    h[...] = arr
    assert h.view(flow=True)[:, 1:11] == approx(arr)

    arr = np.full((12, 12), default)
    h[...] = arr
    assert h.view(flow=True) == approx(arr)

    arr = np.full((9, 1), default)
    with pytest.raises(ValueError):
        h[...] = arr

    arr = np.full((11, 1), default)
    with pytest.raises(ValueError):
        h[...] = arr

    arr = np.full((13, 1), default)
    with pytest.raises(ValueError):
        h[...] = arr

    with pytest.raises(ValueError):
        h[...] = 1

    with pytest.raises(ValueError):
        h[1, 1] = 1


def test_set_array_slice_mismatch_not_at_start():
    h = bh.Histogram(
        bh.axis.Regular(3, 0, 3), bh.axis.Regular(4, 0, 4), bh.axis.Regular(5, 0, 5)
    )

    with pytest.raises(
        ValueError, match=r"Mismatched shapes \(4, 6\) in dimension 2, 6 != 5"
    ):
        h[0, :, :] = np.ones((4, 6))


def test_vectorized_set_basic():
    """NumPy integer-array indices scatter values through the buffer."""
    h = bh.Histogram(
        bh.axis.IntCategory(list(range(5))),
        bh.axis.IntCategory(list(range(6))),
        bh.axis.Regular(7, 0, 7),
    )

    i0 = np.array([0, 2, 4])
    i1 = np.array([1, 3, 5])

    h[i0, i1, :] = 9.0
    assert h.view()[i0, i1, :] == approx(9.0)

    # Untouched cells remain zero
    assert h.view()[1, 0, 0] == 0.0

    # Per-cell values, broadcasting over the trailing slice
    vals = np.arange(3 * 7).reshape(3, 7).astype(float)
    h[i0, i1, :] = vals
    assert h.view()[i0, i1, :] == approx(vals)


def test_vectorized_set_accumulator():
    h = bh.Histogram(bh.axis.Regular(5, 0, 5), storage=bh.storage.Weight())

    idx = np.array([1, 3])
    # The View accepts a trailing (value, variance) dimension
    h[idx] = np.array([[10.0, 1.0], [20.0, 2.0]])
    assert h.view()[idx].value == approx([10.0, 20.0])
    assert h.view()[idx].variance == approx([1.0, 2.0])


def test_vectorized_set_multicell():
    h = bh.Histogram(
        bh.axis.Regular(4, 0, 4),
        bh.axis.Regular(4, 0, 4),
        storage=bh.storage.MultiCell(3),
    )

    i0 = np.array([1, 2])
    i1 = np.array([0, 3])
    h[i0, i1] = 7.0
    assert h.view()[:, i0, i1] == approx(7.0)


def test_set_histogram_with_flow():
    # Issue #1143 (B1): np.asarray(value) dropped a Histogram's flow bins
    h = bh.Histogram(bh.axis.Regular(3, 0, 1))
    h2 = bh.Histogram(bh.axis.Regular(3, 0, 1))
    h2.view(flow=True)[:] = [100, 1, 2, 3, 200]

    h[:] = h2
    assert h.view(flow=True) == approx([100, 1, 2, 3, 200])


def test_set_histogram_with_flow_2d():
    h = bh.Histogram(bh.axis.Regular(2, 0, 1), bh.axis.Regular(2, 0, 1))
    h2 = bh.Histogram(bh.axis.Regular(2, 0, 1), bh.axis.Regular(2, 0, 1))
    h2.view(flow=True)[...] = np.arange(16).reshape(4, 4)

    h[:, :] = h2
    assert h.view(flow=True) == approx(np.arange(16).reshape(4, 4))


def test_set_histogram_without_flow():
    # A flow-less histogram value still sets the inner bins only
    h = bh.Histogram(bh.axis.Regular(3, 0, 1))
    h3 = bh.Histogram(bh.axis.Regular(3, 0, 1, underflow=False, overflow=False))
    h3[:] = [1, 2, 3]

    h[0:3] = h3
    assert h.view(flow=True) == approx([0, 1, 2, 3, 0])


def test_set_histogram_flow_mismatch():
    # Cannot set flow bins of a slice that does not include them
    h = bh.Histogram(bh.axis.Regular(3, 0, 1))
    h2 = bh.Histogram(bh.axis.Regular(3, 0, 1))
    h2.view(flow=True)[:] = [100, 1, 2, 3, 200]

    with pytest.raises(ValueError):
        h[0:3] = h2
