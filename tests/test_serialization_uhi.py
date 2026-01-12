from __future__ import annotations

import numpy as np
import pytest

import boost_histogram as bh
from boost_histogram.serialization import from_uhi, remove_writer_info, to_uhi


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
    data = to_uhi(h)

    assert data["axes"][0]["type"] == "regular"
    assert data["axes"][0]["lower"] == 0
    assert data["axes"][0]["upper"] == 1
    assert data["axes"][0]["bins"] == 10
    assert data["axes"][0]["underflow"]
    assert data["axes"][0]["overflow"]
    assert not data["axes"][0]["circular"]
    assert data["storage"]["type"] == expected_type
    assert data["storage"]["values"] == pytest.approx(np.zeros(12))
    assert data["uhi_schema"] == 1


def test_weighed_to_dict() -> None:
    h = bh.Histogram(
        bh.axis.Integer(3, 15),
        storage=bh.storage.Weight(),
    )
    data = to_uhi(h)

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
    )
    h.name = "hi"
    data = to_uhi(h)

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
    data = to_uhi(h)

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
    data = to_uhi(h)

    assert data["axes"][0]["type"] == "variable"
    assert data["axes"][0]["edges"] == pytest.approx(
        np.exp(np.linspace(0, np.log(10), 11))
    )


def test_transform_sqrt_axis_to_dict() -> None:
    h = bh.Histogram(bh.axis.Regular(10, 0, 10, transform=bh.axis.transform.sqrt))
    data = to_uhi(h)

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
    data = to_uhi(h)
    h2 = from_uhi(data)

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
    data = to_uhi(h)
    h2 = from_uhi(data)

    assert pytest.approx(np.array(h.axes[0])) == np.array(h2.axes[0])
    assert np.asarray(h) == pytest.approx(h2)


def test_round_trip_mean() -> None:
    h = bh.Histogram(
        bh.axis.StrCategory(["1", "2", "3"]),
        storage=bh.storage.Mean(),
    )
    h.fill(["1", "2", "3"], weight=[10, 20, 30], sample=[100, 200, 300])
    h.fill(["1", "2", "3"], weight=[10, 20, 30], sample=[100, 200, 300])
    data = to_uhi(h)
    h2 = from_uhi(data)

    assert pytest.approx(np.array(h.axes[0])) == np.array(h2.axes[0])
    assert np.asarray(h) == pytest.approx(h2)


def test_round_trip_weighted_mean() -> None:
    h = bh.Histogram(
        bh.axis.IntCategory([1, 2, 3]),
        storage=bh.storage.WeightedMean(),
    )
    h.fill([1, 2, 3], weight=[10, 20, 30], sample=[100, 200, 300])
    h.fill([1, 2, 3], weight=[10, 20, 30], sample=[100, 200, 300])
    data = to_uhi(h)
    h2 = from_uhi(data)

    assert pytest.approx(np.array(h.axes[0])) == np.array(h2.axes[0])
    assert np.asarray(h) == pytest.approx(h2)


def test_uhi_wrapper():
    h = bh.Histogram(
        bh.axis.IntCategory([1, 2, 3]),
        storage=bh.storage.WeightedMean(),
    )
    assert to_uhi(h).keys() == h._to_uhi_().keys()
    data = h._to_uhi_()
    assert repr(from_uhi(data)) == repr(bh.Histogram._from_uhi_(data))


def test_uhi_direct_conversion():
    h = bh.Histogram(
        bh.axis.IntCategory([1, 2, 3]),
        storage=bh.storage.Int64(),
    )
    uhi_dict = h._to_uhi_()
    h2 = bh.Histogram(uhi_dict)
    assert h == h2


def test_round_trip_native() -> None:
    h = bh.Histogram(
        bh.axis.Integer(0, 10),
        storage=bh.storage.AtomicInt64(),
    )
    h.fill([-1, 0, 0, 1, 20, 20, 20])
    data = to_uhi(h)
    h2 = from_uhi(data)

    assert h == h2

    assert isinstance(h2.axes[0], bh.axis.Integer)
    assert h2.storage_type is bh.storage.AtomicInt64


@pytest.mark.parametrize("remove", ["boost-histogram", None])
def test_round_trip_clean(remove: str | None) -> None:
    h = bh.Histogram(
        bh.axis.Integer(0, 10),
        storage=bh.storage.AtomicInt64(),
    )
    h.fill([-1, 0, 0, 1, 20, 20, 20])

    data = to_uhi(h)
    data = remove_writer_info(data, library=remove)
    h2 = from_uhi(data)

    assert isinstance(h2.axes[0], bh.axis.Regular)
    assert h2.storage_type is bh.storage.Int64


def test_unserializable_metadata() -> None:
    h = bh.Histogram(
        bh.axis.Integer(0, 10, __dict__={"c": 3, "@d": 4}),
    )
    h.__dict__["a"] = 1
    h.__dict__["@b"] = 2
    data = to_uhi(h)

    assert data["metadata"] == {"a": 1, "_variance_known": True}
    assert data["axes"][0]["metadata"] == {"c": 3}


def test_histogram_metadata() -> None:
    h = bh.Histogram(bh.axis.Integer(0, 10))
    h.name = "Hi"
    h.label = "hi"
    h.other = 3

    data = to_uhi(h)

    assert data["metadata"] == {
        "name": "Hi",
        "label": "hi",
        "other": 3,
        "_variance_known": True,
    }


def test_remove_writer_info() -> None:
    d = {
        "uhi_schema": 1,
        "writer_info": {"boost-histogram": {"foo": "bar"}, "hist": {"FOO": "BAR"}},
    }

    assert remove_writer_info(d, library=None) == {"uhi_schema": 1}
    assert remove_writer_info(d) == {
        "uhi_schema": 1,
        "writer_info": {"hist": {"FOO": "BAR"}},
    }
    assert remove_writer_info(d, library="boost-histogram") == {
        "uhi_schema": 1,
        "writer_info": {"hist": {"FOO": "BAR"}},
    }
    assert remove_writer_info(d, library="hist") == {
        "uhi_schema": 1,
        "writer_info": {"boost-histogram": {"foo": "bar"}},
    }
    assert remove_writer_info(d, library="c") == d


def test_convert_weight() -> None:
    h = bh.Histogram(
        bh.axis.Regular(3, 13, 10, __dict__={"name": "x"}),
        bh.axis.StrCategory(["one", "two"]),
        storage=bh.storage.Weight(),
    )
    data = h._to_uhi_()
    h2 = bh.Histogram(data)

    assert h == h2


def test_convert_weightmean() -> None:
    h = bh.Histogram(
        bh.axis.Regular(12, 0, 1),
        bh.axis.StrCategory(["a", "b", "c", "d", "e", "f", "g"]),
        bh.axis.Boolean(),
        bh.axis.Integer(1, 18),
        storage=bh.storage.WeightedMean(),
    )
    data = h._to_uhi_()
    h2 = bh.Histogram(data)

    assert h.axes == h2.axes
