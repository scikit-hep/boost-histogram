from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import boost_histogram as bh

h5py = pytest.importorskip("h5py")
s = pytest.importorskip("boost_histogram.serialization.hdf5")


@pytest.mark.parametrize(
    ("storage_type", "fill_args", "fill_kwargs"),
    [
        pytest.param(bh.storage.Double(), [0.3, 0.3, 0.4, 1.2], {}, id="double"),
        pytest.param(bh.storage.Int64(), [0.3, 0.3, 0.4, 1.2], {}, id="int64"),
        pytest.param(
            bh.storage.AtomicInt64(), [0.3, 0.3, 0.4, 1.2], {}, id="atomicint"
        ),
        pytest.param(bh.storage.Unlimited(), [0.3, 0.3, 0.4, 1.2], {}, id="unlimited"),
        pytest.param(bh.storage.Weight(), [0.3, 0.3, 0.4, 1.2], {}, id="weight"),
        pytest.param(
            bh.storage.Mean(),
            [0.3, 0.3, 0.4, 1.2, 1.6],
            {"sample": [1, 2, 3, 4, 4]},
            id="mean",
        ),
        pytest.param(
            bh.storage.WeightedMean(),
            [0.3, 0.3, 0.4, 1.2, 1.6],
            {"sample": [1, 2, 3, 4, 4], "weight": [1, 1, 1, 1, 2]},
            id="weighted_mean",
        ),
    ],
)
def test_hdf5_storage(
    tmp_path: Path,
    storage_type: bh.storage.Storage,
    fill_args: list[float],
    fill_kwargs: dict[str, Any],
) -> None:
    h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=storage_type)
    h.fill(fill_args, **fill_kwargs)

    filepath = tmp_path / "hist.h5"
    with h5py.File(filepath, "x") as f:
        grp = f.create_group("test_hist")
        s.write_hdf5_schema(grp, h)

    with h5py.File(filepath) as f:
        re_constructed_hist = s.read_hdf5_schema(f["test_hist"])

    actual_hist = h.copy()

    # checking types of the reconstructed axes
    assert type(actual_hist.axes[0]) is type(re_constructed_hist.axes[0])

    if isinstance(storage_type, bh.storage.Unlimited):
        actual_hist_storage = bh.storage.Double()
    elif isinstance(storage_type, bh.storage.AtomicInt64):
        actual_hist_storage = bh.storage.Int64()
    else:
        actual_hist_storage = storage_type
    assert isinstance(actual_hist_storage, re_constructed_hist.storage_type)

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

    # checking variance or counts if applicable
    if isinstance(storage_type, bh.storage.Weight):
        assert actual_hist.variances() == pytest.approx(
            re_constructed_hist.variances(), abs=1e-4, rel=1e-9
        )
    if isinstance(storage_type, (bh.storage.WeightedMean, bh.storage.Mean)):
        assert actual_hist.counts() == pytest.approx(
            re_constructed_hist.counts(), abs=1e-4, rel=1e-9
        )


def test_hdf5_2d(tmp_path: Path) -> None:
    h = bh.Histogram(bh.axis.Integer(0, 4), bh.axis.StrCategory(["a", "b", "c"]))
    h.fill([0, 1, 1, 1], ["a", "b", "b", "c"])

    filepath = tmp_path / "hist.h5"
    with h5py.File(filepath, "x") as f:
        grp = f.create_group("test_hist")
        s.write_hdf5_schema(grp, h)

    with h5py.File(filepath) as f:
        re_constructed_hist = s.read_hdf5_schema(f["test_hist"])

    actual_hist = h.copy()

    assert isinstance(re_constructed_hist.axes[0], bh.axis.Regular)
    assert type(actual_hist.axes[1]) is type(re_constructed_hist.axes[1])

    assert (
        actual_hist.axes[0].traits.underflow
        == re_constructed_hist.axes[0].traits.underflow
    )
    assert (
        actual_hist.axes[0].traits.overflow
        == re_constructed_hist.axes[0].traits.overflow
    )
    assert (
        actual_hist.axes[0].traits.circular
        == re_constructed_hist.axes[0].traits.circular
    )
    assert actual_hist.axes[1].traits == re_constructed_hist.axes[1].traits

    assert np.asarray(actual_hist.axes[0].edges) == pytest.approx(
        np.asarray(re_constructed_hist.axes[0].edges)
    )
    assert list(actual_hist.axes[1]) == list(re_constructed_hist.axes[1])

    assert h.values() == pytest.approx(re_constructed_hist.values())
    assert h.values(flow=True) == pytest.approx(re_constructed_hist.values(flow=True))
