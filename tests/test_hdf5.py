from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import boost_histogram as bh

s = pytest.importorskip("boost_histogram.serialization.hdf5")


def test_weighted_storge(tmp_path: Path) -> None:
    h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.Weight())
    h.fill([0.3, 0.3, 0.4, 1.2])

    s.write_hdf5_schema(tmp_path / "test_weighted_storage.h5", {"test_hist": h})

    h_constructed = s.read_hdf5_schema(tmp_path / "test_weighted_storage.h5")

    assert {"test_hist"} == h_constructed.keys()

    actual_hist = h.copy()
    re_constructed_hist = h_constructed["test_hist"]

    # checking types of the reconstructed axes
    assert type(actual_hist.axes[0]) is type(re_constructed_hist.axes[0])
    assert actual_hist.storage_type == re_constructed_hist.storage_type
    # checking values of the essential inputs of the axes
    assert actual_hist.axes[0].traits == re_constructed_hist.axes[0].traits
    assert np.allclose(
        actual_hist.axes[0].centers,
        re_constructed_hist.axes[0].centers,
        atol=1e-4,
        rtol=1e-9,
    )
    # checking storage values
    assert np.allclose(
        actual_hist.values(), re_constructed_hist.values(), atol=1e-4, rtol=1e-9
    )
    # checking variance variances
    variances = re_constructed_hist.variances()
    assert variances is not None
    assert np.allclose(actual_hist.variances(), variances, atol=1e-4, rtol=1e-9)


def test_weighted_mean_storage(tmp_path: Path) -> None:
    h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.WeightedMean())
    h.fill([0.3, 0.3, 0.4, 1.2, 1.6], sample=[1, 2, 3, 4, 4], weight=[1, 1, 1, 1, 2])

    s.write_hdf5_schema(tmp_path / "test_weighted_mean_storage.h5", {"test_hist": h})

    h_constructed = s.read_hdf5_schema(tmp_path / "test_weighted_mean_storage.h5")

    assert {"test_hist"} == h_constructed.keys()

    actual_hist = h.copy()
    re_constructed_hist = h_constructed["test_hist"]

    # checking types of the reconstructed axes
    assert type(actual_hist.axes[0]) is type(re_constructed_hist.axes[0])
    assert actual_hist.storage_type == re_constructed_hist.storage_type
    # checking values of the essential inputs of the axes
    assert actual_hist.axes[0].traits == re_constructed_hist.axes[0].traits
    assert np.allclose(
        actual_hist.axes[0].centers,
        re_constructed_hist.axes[0].centers,
        atol=1e-4,
        rtol=1e-9,
    )
    # checking storage values
    assert np.allclose(
        actual_hist.values(), re_constructed_hist.values(), atol=1e-4, rtol=1e-9
    )
    # checking variance variances
    print(actual_hist.view(), re_constructed_hist.view())
    print(actual_hist.variances())
    # assert np.allclose(actual_hist.variances(), re_constructed_hist.variances(), atol=1e-4, rtol=1e-9)


def test_mean_storage(tmp_path: Path) -> None:
    h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.Mean())
    h.fill([0.3, 0.3, 0.4, 1.2, 1.6], sample=[1, 2, 3, 4, 4])

    s.write_hdf5_schema(tmp_path / "test_mean_storage.h5", {"test_hist": h})

    h_constructed = s.read_hdf5_schema(tmp_path / "test_mean_storage.h5")

    assert {"test_hist"} == h_constructed.keys()

    actual_hist = h.copy()
    re_constructed_hist = h_constructed["test_hist"]

    # checking types of the reconstructed axes
    assert type(actual_hist.axes[0]) is type(re_constructed_hist.axes[0])
    assert actual_hist.storage_type == re_constructed_hist.storage_type
    # checking values of the essential inputs of the axes
    assert actual_hist.axes[0].traits == re_constructed_hist.axes[0].traits
    assert np.allclose(
        actual_hist.axes[0].centers,
        re_constructed_hist.axes[0].centers,
        atol=1e-4,
        rtol=1e-9,
    )
    # checking storage values
    assert np.allclose(
        actual_hist.values(), re_constructed_hist.values(), atol=1e-4, rtol=1e-9
    )
    # checking variance variances
    # assert np.allclose(actual_hist.variances(), re_constructed_hist.variances(), atol=1e-4, rtol=1e-9)
    assert np.allclose(
        actual_hist.counts(), re_constructed_hist.counts(), atol=1e-4, rtol=1e-9
    )
