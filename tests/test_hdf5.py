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
        pytest.param(bh.storage.Weight(), [0.3, 0.3, 0.4, 1.2], {}, id="weight"),
        pytest.param(
            bh.storage.WeightedMean(),
            [0.3, 0.3, 0.4, 1.2, 1.6],
            {"sample": [1, 2, 3, 4, 4], "weight": [1, 1, 1, 1, 2]},
            id="weighted_mean",
        ),
        pytest.param(
            bh.storage.Mean(),
            [0.3, 0.3, 0.4, 1.2, 1.6],
            {"sample": [1, 2, 3, 4, 4]},
            id="mean",
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

    # checking variance or counts if applicable
    if isinstance(storage_type, bh.storage.Weight):
        assert actual_hist.variances() == pytest.approx(
            re_constructed_hist.variances(), abs=1e-4, rel=1e-9
        )
    if isinstance(storage_type, (bh.storage.WeightedMean, bh.storage.Mean)):
        assert actual_hist.counts() == pytest.approx(
            re_constructed_hist.counts(), abs=1e-4, rel=1e-9
        )
