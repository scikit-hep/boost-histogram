from __future__ import annotations

import numpy as np
import pytest

import boost_histogram as bh
from boost_histogram.serialization import generic


@pytest.mark.parametrize(
    ("storage_type", "expected_type"),
    [
        pytest.param(bh.storage.AtomicInt64(), "int", id="atomic_int"),
        pytest.param(bh.storage.Int64(), "int", id="int"),
        pytest.param(
            bh.storage.Unlimited(), "double", id="unlimited"
        ),  # This always renders as double
        pytest.param(bh.storage.Double(), "double", id="double"),
    ],
)
def test_simple_to_dict(storage_type: bh.storage.Storage, expected_type: str) -> None:
    h = bh.Histogram(
        bh.axis.Regular(10, 0, 1),
        storage=storage_type,
    )
    data = generic.to_dict(h)

    assert "metadata" not in data
    assert data["axes"][0]["type"] == "regular"
    assert data["axes"][0]["lower"] == 0
    assert data["axes"][0]["upper"] == 1
    assert data["axes"][0]["bins"] == 10
    assert data["axes"][0]["underflow"]
    assert data["axes"][0]["overflow"]
    assert not data["axes"][0]["circular"]
    assert data["storage"]["type"] == expected_type
    assert data["storage"]["values"] == pytest.approx(np.zeros(12))


def test_weighed_to_dict() -> None:
    h = bh.Histogram(
        bh.axis.Integer(3, 15),
        storage=bh.storage.Weight(),
    )
    data = generic.to_dict(h)

    assert data["axes"][0]["type"] == "regular"
    assert data["axes"][0]["lower"] == 3
    assert data["axes"][0]["upper"] == 15
    assert data["axes"][0]["bins"] == 12
    assert data["axes"][0]["underflow"]
    assert data["axes"][0]["overflow"]
    assert not data["axes"][0]["circular"]
    assert data["storage"]["type"] == "weighted"
    assert data["storage"]["values"] == pytest.approx(np.zeros(14))
    assert data["storage"]["variances"] == pytest.approx(np.zeros(14))


def test_mean_to_dict() -> None:
    h = bh.Histogram(
        bh.axis.StrCategory(["one", "two", "three"]),
        storage=bh.storage.Mean(),
        metadata={"name": "hi"},
    )
    data = generic.to_dict(h)

    assert data["metadata"]["name"] == "hi"
    assert data["axes"][0]["type"] == "category_str"
    assert data["axes"][0]["categories"] == ["one", "two", "three"]
    assert data["axes"][0]["flow"]
    assert data["storage"]["type"] == "mean"
    assert data["storage"]["counts"] == pytest.approx(np.zeros(4))
    assert data["storage"]["values"] == pytest.approx(np.zeros(4))
    assert data["storage"]["variances"] == pytest.approx(np.zeros(4))


def test_weighted_mean_to_dict() -> None:
    h = bh.Histogram(
        bh.axis.IntCategory([1, 2, 3]),
        storage=bh.storage.WeightedMean(),
    )
    h.fill([1, 2, 3, 50], weight=[10, 20, 30, 5], sample=[100, 200, 300, 1])
    h.fill([1, 2, 3, -3], weight=[10, 20, 30, 5], sample=[100, 200, 300, 1])
    data = generic.to_dict(h)

    assert data["axes"][0]["type"] == "category_int"
    assert data["axes"][0]["categories"] == pytest.approx([1, 2, 3])
    assert data["axes"][0]["flow"]
    assert data["storage"]["type"] == "weighted_mean"
    assert data["storage"]["sum_of_weights"] == pytest.approx(
        np.array([20, 40, 60, 10])
    )
    assert data["storage"]["sum_of_weights_squared"] == pytest.approx(
        np.array([200, 800, 1800, 50])
    )
    assert data["storage"]["values"] == pytest.approx(np.array([100, 200, 300, 1]))
    assert data["storage"]["variances"] == pytest.approx(np.zeros(4))


def test_transform_log_axis_to_dict() -> None:
    h = bh.Histogram(bh.axis.Regular(10, 1, 10, transform=bh.axis.transform.log))
    data = generic.to_dict(h)

    assert data["axes"][0]["type"] == "variable"
    assert data["axes"][0]["edges"] == pytest.approx(
        np.exp(np.linspace(0, np.log(10), 11))
    )


def test_transform_sqrt_axis_to_dict() -> None:
    h = bh.Histogram(bh.axis.Regular(10, 0, 10, transform=bh.axis.transform.sqrt))
    data = generic.to_dict(h)

    assert data["axes"][0]["type"] == "variable"
    assert data["axes"][0]["edges"] == pytest.approx(
        (np.linspace(0, np.sqrt(10), 11)) ** 2
    )


@pytest.mark.parametrize(
    "storage_type",
    [
        pytest.param(bh.storage.AtomicInt64(), id="atomic_int"),
        pytest.param(bh.storage.Int64(), id="int"),
        pytest.param(bh.storage.Double(), id="double"),
        pytest.param(bh.storage.Unlimited(), id="unlimited"),
    ],
)
def test_round_trip_simple(storage_type: bh.storage.Storage) -> None:
    h = bh.Histogram(
        bh.axis.Regular(10, 0, 10),
        storage=storage_type,
    )
    h.fill([-1, 0, 0, 1, 20, 20, 20])
    data = generic.to_dict(h)
    h2 = generic.from_dict(data)

    if isinstance(storage_type, (bh.storage.Int64, bh.storage.Double)):
        assert h == h2

    assert h.view() == pytest.approx(h2.view())


def test_round_trip_weighted() -> None:
    h = bh.Histogram(
        bh.axis.Variable([1, 2, 4, 5], circular=True),
        storage=bh.storage.Weight(),
    )
    h.fill(["1", "2", "3"], weight=[10, 20, 30])
    h.fill(["1", "2", "3"], weight=[10, 20, 30])
    data = generic.to_dict(h)
    h2 = generic.from_dict(data)

    print(h.view())
    print(h2.view())

    assert pytest.approx(np.array(h.axes[0])) == np.array(h2.axes[0])
    assert np.asarray(h) == pytest.approx(h2)


def test_round_trip_mean() -> None:
    h = bh.Histogram(
        bh.axis.StrCategory(["1", "2", "3"]),
        storage=bh.storage.Mean(),
    )
    h.fill(["1", "2", "3"], weight=[10, 20, 30], sample=[100, 200, 300])
    h.fill(["1", "2", "3"], weight=[10, 20, 30], sample=[100, 200, 300])
    data = generic.to_dict(h)
    h2 = generic.from_dict(data)

    assert pytest.approx(np.array(h.axes[0])) == np.array(h2.axes[0])
    assert np.asarray(h) == pytest.approx(h2)


def test_round_trip_weighted_mean() -> None:
    h = bh.Histogram(
        bh.axis.IntCategory([1, 2, 3]),
        storage=bh.storage.WeightedMean(),
    )
    h.fill([1, 2, 3], weight=[10, 20, 30], sample=[100, 200, 300])
    h.fill([1, 2, 3], weight=[10, 20, 30], sample=[100, 200, 300])
    data = generic.to_dict(h)
    h2 = generic.from_dict(data)

    assert pytest.approx(np.array(h.axes[0])) == np.array(h2.axes[0])
    assert np.asarray(h) == pytest.approx(h2)
