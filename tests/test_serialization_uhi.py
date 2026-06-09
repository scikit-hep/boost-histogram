from __future__ import annotations

import ctypes
import importlib.resources
import json
import math
import sys

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
    writer_info = data["axes"][0]["writer_info"]["boost-histogram"]
    assert writer_info["transform"] == "log"
    assert writer_info["bins"] == 10
    assert writer_info["lower"] == pytest.approx(1)
    assert writer_info["upper"] == pytest.approx(10)


def test_transform_sqrt_axis_to_dict() -> None:
    h = bh.Histogram(bh.axis.Regular(10, 0, 10, transform=bh.axis.transform.sqrt))
    data = to_uhi(h)

    assert data["axes"][0]["type"] == "variable"
    assert data["axes"][0]["edges"] == pytest.approx(
        (np.linspace(0, np.sqrt(10), 11)) ** 2
    )
    writer_info = data["axes"][0]["writer_info"]["boost-histogram"]
    assert writer_info["transform"] == "sqrt"
    assert writer_info["bins"] == 10
    assert writer_info["lower"] == pytest.approx(0)
    assert writer_info["upper"] == pytest.approx(10)


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(bh.axis.transform.log, id="log"),
        pytest.param(bh.axis.transform.sqrt, id="sqrt"),
        pytest.param(bh.axis.transform.Pow(2), id="pow2"),
        pytest.param(bh.axis.transform.Pow(0.5), id="pow_half"),
    ],
)
def test_round_trip_transform(transform: bh.axis.transform.AxisTransform) -> None:
    h = bh.Histogram(bh.axis.Regular(5, 1, 10, transform=transform))
    h.fill([0.5, 1.5, 3.0, 8.0, 50.0])
    data = _json_round_trip(to_uhi(h))
    h2 = from_uhi(data)

    assert isinstance(h2.axes[0], bh.axis.Regular)
    assert h2.axes[0].transform is not None
    assert repr(h2.axes[0].transform) == repr(transform)
    assert h == h2


@pytest.mark.skipif(
    sys.implementation.name == "pypy",
    reason="ctypes function-pointer transforms hang forever on PyPy",
)
@pytest.mark.skipif(
    sys.implementation.name == "graalpy",
    reason="ctypes function-pointer transforms are not supported on GraalPy",
)
def test_round_trip_transform_custom_falls_back_to_variable() -> None:
    """A transform we can't reconstruct (a raw function pointer) degrades to a
    Variable axis rather than failing."""
    ftype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
    transform = bh.axis.transform.Function(
        ftype(math.log), ftype(math.exp), name="custom"
    )
    h = bh.Histogram(bh.axis.Regular(5, 1, 10, transform=transform))
    data = to_uhi(h)

    assert "writer_info" not in data["axes"][0]

    h2 = from_uhi(data)
    assert isinstance(h2.axes[0], bh.axis.Variable)
    assert np.asarray(h2.axes[0].edges) == pytest.approx(np.asarray(h.axes[0].edges))


def test_from_uhi_unknown_transform_falls_back_to_variable() -> None:
    """An unrecognized transform name in writer_info reads back as a Variable
    axis instead of raising."""
    data = {
        "uhi_schema": 1,
        "axes": [
            {
                "type": "variable",
                "edges": [1.0, 2.0, 4.0, 10.0],
                "underflow": True,
                "overflow": True,
                "circular": False,
                "writer_info": {
                    "boost-histogram": {
                        "transform": "mystery",
                        "bins": 3,
                        "lower": 1.0,
                        "upper": 10.0,
                    }
                },
            }
        ],
        "storage": {"type": "double"},
        "metadata": {},
    }

    h = from_uhi(data)
    assert isinstance(h.axes[0], bh.axis.Variable)
    assert np.asarray(h.axes[0].edges) == pytest.approx([1.0, 2.0, 4.0, 10.0])


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


@pytest.mark.parametrize(
    ("storage_type", "expected_type"),
    [
        pytest.param(bh.storage.Int64(), "int", id="int64"),
        pytest.param(bh.storage.AtomicInt64(), "int", id="atomic_int64"),
        pytest.param(bh.storage.Double(), "double", id="double"),
        pytest.param(bh.storage.Weight(), "weighted", id="weight"),
        pytest.param(bh.storage.Mean(), "mean", id="mean"),
    ],
)
def test_to_uhi_keep_storage_option(
    storage_type: bh.storage.Storage, expected_type: str
) -> None:
    h = bh.Histogram(
        bh.axis.Regular(3, 0, 1),
        storage=storage_type,
    )
    data_with = to_uhi(h)
    data_without = to_uhi(h, keep_storage=False)

    assert "storage" in data_with
    assert "storage" in data_without
    # Storage with data has "values" key
    assert "values" in data_with["storage"]
    # Storage without data has only type information
    assert "values" not in data_without["storage"]
    assert data_without["storage"]["type"] == expected_type


@pytest.mark.parametrize(
    "storage_type",
    [
        pytest.param(bh.storage.Int64(), id="int64"),
        pytest.param(bh.storage.AtomicInt64(), id="atomic_int64"),
        pytest.param(bh.storage.Double(), id="double"),
        pytest.param(bh.storage.Weight(), id="weight"),
        pytest.param(bh.storage.Mean(), id="mean"),
    ],
)
def test_from_uhi_missing_storage_data(storage_type: bh.storage.Storage) -> None:
    h = bh.Histogram(
        bh.axis.Regular(4, 0.0, 1.0),
        storage=storage_type,
    )
    # produce a UHI dict with storage type but no data
    data = to_uhi(h, keep_storage=False)

    h2 = from_uhi(data)

    # axes and storage type should round-trip, data should be zeros
    assert pytest.approx(np.array(h.axes[0])) == np.array(h2.axes[0])
    assert h2.storage_type is type(storage_type)
    assert np.asarray(h2) == pytest.approx(np.zeros_like(np.asarray(h2)))


@pytest.mark.parametrize(
    ("storage_type", "expected_type"),
    [
        pytest.param(bh.storage.AtomicInt64(), "int", id="atomic_int64"),
        pytest.param(bh.storage.Unlimited(), "double", id="unlimited"),
    ],
)
def test_from_uhi_old_style_writer_info(
    storage_type: bh.storage.Storage, expected_type: str
) -> None:
    h = bh.Histogram(
        bh.axis.Regular(4, 0.0, 1.0),
        storage=storage_type,
    )
    h.fill([0, 0, 1, 2])

    data = to_uhi(h)

    old_style_data = {
        "uhi_schema": 1,
        "axes": data["axes"],
        "storage": {
            "type": expected_type,
            "values": data["storage"]["values"],
            "writer_info": {
                "boost-histogram": {"orig_type": type(storage_type).__name__}
            },
        },
        "metadata": data.get("metadata", {}),
    }

    h2 = from_uhi(old_style_data)

    assert h == h2
    assert h2.storage_type is type(storage_type)


@pytest.mark.parametrize(
    ("axis_type", "axis_args", "fill_values"),
    [
        pytest.param(bh.axis.Integer, (0, 10), [-1, 0, 0, 1, 20, 20, 20], id="integer"),
        pytest.param(
            bh.axis.Regular,
            (10, 0.0, 10.0),
            [-1.0, 0.0, 0.0, 1.0, 20.0, 20.0, 20.0],
            id="regular",
        ),
        pytest.param(
            bh.axis.Variable,
            ([0.0, 2.0, 5.0, 10.0],),
            [-1.0, 0.0, 1.0, 6.0, 20.0, 20.0],
            id="variable",
        ),
        pytest.param(
            bh.axis.IntCategory,
            ([1, 2, 3],),
            [0, 1, 1, 2, 10, 10, 10],
            id="intcategory",
        ),
        pytest.param(
            bh.axis.StrCategory,
            (["A", "B", "C"],),
            ["Z", "A", "A", "B", "X", "X", "X"],
            id="strcategory",
        ),
    ],
)
def test_round_trip_growth(
    axis_type: type, axis_args: tuple, fill_values: list
) -> None:
    h = bh.Histogram(
        axis_type(*axis_args, growth=True),
    )
    h.fill(fill_values)
    data = to_uhi(h)
    h2 = from_uhi(data)

    assert h == h2

    assert isinstance(h2.axes[0], axis_type)
    assert h2.axes[0].traits.growth == h.axes[0].traits.growth


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


def _json_round_trip(data: dict) -> dict:
    """Simulate a JSON round-trip, which can collapse empty array dimensions."""
    return json.loads(
        json.dumps(
            data,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
        )
    )


@pytest.mark.parametrize(
    "storage_type",
    [
        pytest.param(bh.storage.Double(), id="double"),
        pytest.param(bh.storage.Int64(), id="int64"),
        pytest.param(bh.storage.Weight(), id="weight"),
        pytest.param(bh.storage.Mean(), id="mean"),
        pytest.param(bh.storage.WeightedMean(), id="weighted_mean"),
    ],
)
def test_round_trip_3d_histogram_json(storage_type: bh.storage.Storage) -> None:
    """Regression test: 3D histograms with empty StrCategory axes round-trip
    correctly through JSON serialization.

    JSON serialization collapses empty array dimensions (e.g. shape (5, 0, 0)
    becomes (5, 0)), which must be handled correctly during deserialization.
    """
    h = bh.Histogram(
        bh.axis.Variable([0, 1, 2, 3]),
        bh.axis.StrCategory([], growth=True),
        bh.axis.StrCategory([], growth=True),
        storage=storage_type,
    )
    data = to_uhi(h)
    data = _json_round_trip(data)
    h2 = from_uhi(data)

    assert h.ndim == h2.ndim
    assert h == h2


def test_round_trip_3d_histogram_json_constructor() -> None:
    """Regression test: bh.Histogram(uhi_dict) works for 3D histograms after
    JSON round-trip (reproduces the exact example from the bug report)."""
    h = bh.Histogram(
        bh.axis.Variable([0, 1, 2, 3]),
        bh.axis.StrCategory([], growth=True),
        bh.axis.StrCategory([], growth=True),
        storage=bh.storage.Weight(),
    )
    data = _json_round_trip(to_uhi(h))
    h2 = bh.Histogram(data)

    assert h.ndim == h2.ndim
    assert h == h2


def test_from_uhi_malformed_weight_storage() -> None:
    """Test that malformed Weight storage ( missing required keys) raises ValueError."""
    data = {
        "uhi_schema": 1,
        "axes": [
            {
                "type": "regular",
                "lower": 0,
                "upper": 10,
                "bins": 5,
                "underflow": True,
                "overflow": True,
                "circular": False,
            }
        ],
        "storage": {
            "type": "weighted",
            "values": [1, 2, 3, 4, 5],
        },
        "metadata": {},
    }

    with pytest.raises(ValueError, match="Weighted storage missing required keys"):
        from_uhi(data)


def test_from_uhi_malformed_mean_storage() -> None:
    """Test that malformed Mean storage (missing required keys) raises ValueError."""
    data = {
        "uhi_schema": 1,
        "axes": [
            {
                "type": "regular",
                "lower": 0,
                "upper": 10,
                "bins": 5,
                "underflow": True,
                "overflow": True,
                "circular": False,
            }
        ],
        "storage": {
            "type": "mean",
            "counts": [1, 2, 3, 4, 5],
            "values": [1, 2, 3, 4, 5],
        },
        "metadata": {},
    }

    with pytest.raises(ValueError, match="Mean storage missing required keys"):
        from_uhi(data)


def test_from_uhi_malformed_weighted_mean_storage() -> None:
    """Test that malformed WeightedMean storage (missing required keys) raises ValueError."""
    data = {
        "uhi_schema": 1,
        "axes": [
            {
                "type": "regular",
                "lower": 0,
                "upper": 10,
                "bins": 5,
                "underflow": True,
                "overflow": True,
                "circular": False,
            }
        ],
        "storage": {
            "type": "weighted_mean",
            "sum_of_weights": [1, 2, 3, 4, 5],
            "values": [1, 2, 3, 4, 5],
        },
        "metadata": {},
    }

    with pytest.raises(ValueError, match="Weighted_mean storage missing required keys"):
        from_uhi(data)


def test_from_uhi_structure_only_no_error() -> None:
    """Test that structure-only (no data keys) histograms load correctly."""
    data = {
        "uhi_schema": 1,
        "axes": [
            {
                "type": "regular",
                "lower": 0,
                "upper": 10,
                "bins": 5,
                "underflow": True,
                "overflow": True,
                "circular": False,
            }
        ],
        "storage": {
            "type": "double",
        },
        "metadata": {},
    }

    h = from_uhi(data)
    assert h.storage_type is bh.storage.Double
    assert np.asarray(h) == pytest.approx(np.zeros(5))


@pytest.mark.parametrize(
    "storage_type",
    [
        pytest.param(bh.storage.Int64(), id="int64"),
        pytest.param(bh.storage.AtomicInt64(), id="atomic_int64"),
        pytest.param(bh.storage.Double(), id="double"),
        pytest.param(bh.storage.Unlimited(), id="unlimited"),
        pytest.param(bh.storage.Weight(), id="weight"),
        pytest.param(bh.storage.Mean(), id="mean"),
        pytest.param(bh.storage.WeightedMean(), id="weighted_mean"),
    ],
)
@pytest.mark.parametrize(
    "keep_storage", [True, False], ids=["with_data", "metadata_only"]
)
def test_to_uhi_matches_uhi_schema(
    storage_type: bh.storage.Storage, keep_storage: bool
) -> None:
    """``to_uhi`` output conforms to the UHI histogram JSON schema, including the
    metadata-only (``keep_storage=False``) form used for type-only storage."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        importlib.resources.files("uhi.resources")
        .joinpath("histogram.schema.json")
        .read_text()
    )

    h = bh.Histogram(
        bh.axis.Regular(3, 0, 1),
        bh.axis.IntCategory([1, 2]),
        storage=storage_type,
    )
    if isinstance(storage_type, (bh.storage.Mean, bh.storage.WeightedMean)):
        h.fill([0.1, 0.5, 0.5], [1, 2, 2], sample=[1.0, 2.0, 3.0])
    else:
        h.fill([0.1, 0.5, 0.5], [1, 2, 2])

    data = _json_round_trip(to_uhi(h, keep_storage=keep_storage))

    # The schema describes a mapping of named histograms.
    jsonschema.validate({"hist": data}, schema)
