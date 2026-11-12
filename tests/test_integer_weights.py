from __future__ import annotations

import numpy as np
import pytest

import boost_histogram as bh

STORAGES = [bh.storage.Int64, bh.storage.AtomicInt64]
WIDE = (1 << 53) + 1
MINIMUM = -(1 << 63)
MAXIMUM = (1 << 63) - 1


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("value", [WIDE, -WIDE, MINIMUM, MAXIMUM])
@pytest.mark.parametrize("form", ["scalar", "numpy", "zero_dim", "list", "array"])
def test_signed_integer_weight(storage, value, form):
    weights = {
        "scalar": value,
        "numpy": np.int64(value),
        "zero_dim": np.array(value, dtype=np.int64),
        "list": [value],
        "array": np.array([value], dtype=np.int64),
    }
    h = bh.Histogram(bh.axis.Integer(0, 1), storage=storage())
    h.fill(0, weight=weights[form])
    assert h[0] == value


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("value", [WIDE, MAXIMUM])
@pytest.mark.parametrize("form", ["numpy", "zero_dim", "list", "array"])
def test_representable_unsigned_integer_weight(storage, value, form):
    weights = {
        "numpy": np.uint64(value),
        "zero_dim": np.array(value, dtype=np.uint64),
        "list": [np.uint64(value)],
        "array": np.array([value], dtype=np.uint64),
    }
    h = bh.Histogram(bh.axis.Integer(0, 1), storage=storage())
    h.fill(0, weight=weights[form])
    assert h[0] == value


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("reverse", [False, True])
def test_integer_weight_strides(storage, dtype, reverse):
    original = np.array([WIDE, 4, MAXIMUM, 5, 3, 6], dtype=dtype)
    weights = original[::2]
    if reverse:
        weights = weights[::-1]
    assert not weights.flags.c_contiguous
    h = bh.Histogram(bh.axis.Integer(0, 3), storage=storage())
    h.fill(np.arange(3), weight=weights)
    np.testing.assert_array_equal(h.view(), weights)


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("weight", [WIDE, np.array(WIDE, dtype=np.int64)])
def test_integer_scalar_broadcast(storage, weight):
    h = bh.Histogram(bh.axis.Integer(0, 2), storage=storage())
    h.fill([0, 1, 0, 1], weight=weight)
    np.testing.assert_array_equal(h.view(), [2 * WIDE, 2 * WIDE])


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_threaded_integer_weights(storage, dtype):
    # Keep each bin's sum representable; this tests ingestion, not overflow.
    values = (
        [MAXIMUM, WIDE, 0, 3] if dtype is np.uint64 else [MAXIMUM, MINIMUM, WIDE, -WIDE]
    )
    weights = np.array(values, dtype=dtype)
    h = bh.Histogram(bh.axis.Integer(0, 4), storage=storage())
    h.fill(np.arange(4), weight=weights, threads=2)
    np.testing.assert_array_equal(h.view(), weights)


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("export", [[], [7, 8]])
def test_raw_numeric_array_preserves_scalar_conversion(storage, export):
    array_calls = []

    class NumericArray:
        def __float__(self):
            return 3.0

        def __index__(self):
            return 4

        def __array__(self, dtype=None, copy=None):
            array_calls.append((dtype, copy))
            return np.array(export, dtype=np.int64)

    # An unrelated empty or multi-element array export must not become a scalar
    # buffer dereference. Match the native binding's existing numeric protocol.
    h = bh.Histogram(bh.axis.Integer(0, 1), storage=storage())
    h._hist.fill(0, weight=NumericArray())
    assert h[0] == 3
    assert not array_calls


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("zero_dim", [False, True])
def test_raw_numpy_integer_scalar(storage, dtype, zero_dim):
    value = np.array(WIDE, dtype=dtype) if zero_dim else dtype(WIDE)
    h = bh.Histogram(bh.axis.Integer(0, 1), storage=storage())
    h._hist.fill(0, weight=value)
    assert h[0] == WIDE


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("weight", [True, np.bool_(True), np.array(True)])
def test_boolean_weights(storage, weight):
    h = bh.Histogram(bh.axis.Integer(0, 1), storage=storage())
    h.fill(0, weight=weight)
    assert h[0] == 1


@pytest.mark.parametrize("storage", STORAGES)
def test_empty_integer_weights(storage):
    h = bh.Histogram(bh.axis.Integer(0, 1), storage=storage())
    h.fill(np.array([], dtype=np.int64), weight=np.array([], dtype=np.int64))
    assert h[0] == 0


@pytest.mark.parametrize("storage", STORAGES)
def test_fractional_weights_retain_existing_conversion(storage):
    h = bh.Histogram(bh.axis.Integer(0, 3), storage=storage())
    h.fill([0, 1, 2], weight=[0.5, 1.5, -1.5])
    np.testing.assert_array_equal(h.view(), [0, 1, -1])


@pytest.mark.parametrize("storage", [bh.storage.Double, bh.storage.Unlimited])
def test_other_sum_storages(storage):
    h = bh.Histogram(bh.axis.Integer(0, 1), storage=storage())
    h.fill([0, 0], weight=[2, 3])
    assert h[0] == 5


def test_weight_storage():
    h = bh.Histogram(bh.axis.Integer(0, 1), storage=bh.storage.Weight())
    h.fill([0, 0], weight=[2, 3])
    assert h[0].value == 5
    assert h[0].variance == 13


def test_mean_storage():
    h = bh.Histogram(bh.axis.Integer(0, 1), storage=bh.storage.Mean())
    h.fill([0, 0], sample=[10, 20])
    assert h[0].count == 2
    assert h[0].value == 15
    assert h[0].variance == 50


def test_weighted_mean_storage():
    h = bh.Histogram(bh.axis.Integer(0, 1), storage=bh.storage.WeightedMean())
    h.fill([0, 0], weight=[2, 3], sample=[10, 20])
    assert h[0].sum_of_weights == 5
    assert h[0].sum_of_weights_squared == 13
    assert h[0].value == 16
    assert h[0].variance == 50


def test_multicell_storage():
    h = bh.Histogram(bh.axis.Integer(0, 1), storage=bh.storage.MultiCell(2))
    h.fill([0, 0], weight=[[1, 2], [3, 4]])
    np.testing.assert_array_equal(h[0], [4, 6])
