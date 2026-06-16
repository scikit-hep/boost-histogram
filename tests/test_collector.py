from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import boost_histogram as bh

V = bh.accumulators.Values


def _filled_1d():
    """A 1D Collector histogram where bin i holds the values filled into it."""
    h = bh.Histogram(bh.axis.Regular(3, 0, 3), storage=bh.storage.Collector())
    h.fill([0, 0, 1, 2, 2, 2], sample=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    return h


def test_construction_and_repr():
    h = bh.Histogram(bh.axis.Regular(3, 0, 3), storage=bh.storage.Collector())
    assert isinstance(h.storage_type(), bh.storage.Collector)
    assert repr(bh.storage.Collector()) == "Collector()"
    assert "Collector()" in repr(h)
    # str must not raise for a 1D histogram (no scalar rendering for ragged cells)
    assert "Collector" in str(h)


def test_fill_and_index():
    h = _filled_1d()
    # h[i] is a Values accumulator (the per-bin cell)
    assert isinstance(h[0], V)
    assert list(h[0]) == [1.0, 2.0]
    assert list(h[1]) == [3.0]
    assert list(h[2]) == [4.0, 5.0, 6.0]
    assert h[0][0] == 1.0
    assert len(h[0]) == 2
    assert_array_equal(h[0].value, [1.0, 2.0])


def test_view_is_object_array():
    h = _filled_1d()
    v = h.view()
    assert v.dtype == object
    assert v.shape == (3,)
    assert_array_equal(v[0], [1.0, 2.0])
    assert_array_equal(v[2], [4.0, 5.0, 6.0])
    # each cell is its own float64 array
    assert v[0].dtype == np.float64


def test_view_flow():
    h = bh.Histogram(bh.axis.Regular(2, 0, 2), storage=bh.storage.Collector())
    h.fill([-1, 0, 1, 5], sample=[10.0, 1.0, 2.0, 99.0])
    assert [list(c) for c in h.view()] == [[1.0], [2.0]]
    assert [list(c) for c in h.view(flow=True)] == [[10.0], [1.0], [2.0], [99.0]]


def test_scalar_arg_broadcast():
    h = bh.Histogram(bh.axis.Regular(1, 0, 1), storage=bh.storage.Collector())
    h.fill(0, sample=[1.0, 2.0, 3.0])
    assert list(h[0]) == [1.0, 2.0, 3.0]


def test_2d():
    h = bh.Histogram(
        bh.axis.Regular(2, 0, 2),
        bh.axis.Regular(2, 0, 2),
        storage=bh.storage.Collector(),
    )
    h.fill([0, 0, 1], [0, 1, 1], sample=[1.0, 2.0, 3.0])
    v = h.view()
    assert v.shape == (2, 2)
    assert list(h[0, 0]) == [1.0]
    assert list(h[0, 1]) == [2.0]
    assert list(h[1, 0]) == []
    assert list(h[1, 1]) == [3.0]


def test_sum_is_concatenation():
    h = _filled_1d()
    assert isinstance(h.sum(), V)
    assert sorted(h.sum()) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # empty histogram sums to an empty cell
    empty = bh.Histogram(bh.axis.Regular(3, 0, 3), storage=bh.storage.Collector())
    assert list(empty.sum()) == []


def test_addition_concatenates():
    h = _filled_1d()
    hh = h + h
    assert list(hh[0]) == [1.0, 2.0, 1.0, 2.0]
    assert list(hh[1]) == [3.0, 3.0]


def test_project_concatenates():
    h = bh.Histogram(
        bh.axis.Regular(2, 0, 2),
        bh.axis.Regular(2, 0, 2),
        storage=bh.storage.Collector(),
    )
    h.fill([0, 0, 1], [0, 1, 1], sample=[1.0, 2.0, 3.0])
    p = h.project(0)
    assert list(p[0]) == [1.0, 2.0]
    assert list(p[1]) == [3.0]


def test_slice_reduce():
    h = bh.Histogram(bh.axis.Regular(4, 0, 4), storage=bh.storage.Collector())
    h.fill([0, 1, 2, 3], sample=[1.0, 2.0, 3.0, 4.0])
    sub = h[1:3]
    assert sub.ndim == 1
    assert [list(c) for c in sub.view()] == [[2.0], [3.0]]


def test_factor_rebin_concatenates():
    h = bh.Histogram(bh.axis.Regular(4, 0, 4), storage=bh.storage.Collector())
    h.fill([0, 1, 2, 3], sample=[1.0, 2.0, 3.0, 4.0])
    rebinned = h[:: bh.rebin(2)]
    assert [list(c) for c in rebinned.view()] == [[1.0, 2.0], [3.0, 4.0]]


def test_pickle_roundtrip():
    h = _filled_1d()
    h2 = pickle.loads(pickle.dumps(h))
    assert h2 == h
    assert list(h2[2]) == [4.0, 5.0, 6.0]


@pytest.mark.parametrize(
    "corrupt",
    [
        [-1, 2, 1, 3, 0],  # negative count
        [0, 2, 1, 4, 0],  # counts sum exceeds value buffer
        [0, 2, 1, 2, 0],  # counts sum falls short of value buffer
    ],
)
def test_pickle_rejects_corrupt_counts(corrupt):
    # The on-disk layout is a per-bin int64 counts array plus a flat float64 value
    # array; load() must reject counts that don't describe the value buffer exactly,
    # otherwise native code reads past the NumPy buffer (see PR #1133 review).
    data = pickle.dumps(_filled_1d())
    counts = np.array([0, 2, 1, 3, 0], dtype=np.int64).tobytes()
    assert data.count(counts) == 1
    bad = data.replace(counts, np.array(corrupt, dtype=np.int64).tobytes())
    with pytest.raises(Exception, match="collector"):
        pickle.loads(bad)


def test_copy_and_deepcopy():
    h = _filled_1d()
    assert copy.copy(h) == h
    assert copy.deepcopy(h) == h


def test_reset():
    h = _filled_1d()
    h.reset()
    assert [list(c) for c in h.view()] == [[], [], []]


def test_equality():
    assert _filled_1d() == _filled_1d()
    other = bh.Histogram(bh.axis.Regular(3, 0, 3), storage=bh.storage.Collector())
    other.fill([0], sample=[1.0])
    assert _filled_1d() != other


# --- Unsupported operations (the object view is copy-only) ---


def test_setitem_raises():
    h = _filled_1d()
    with pytest.raises(NotImplementedError):
        h[0] = [1.0, 2.0]


def test_scalar_arithmetic_raises():
    h = _filled_1d()
    with pytest.raises(NotImplementedError):
        h * 2


def test_histogram_subtraction_raises():
    h = _filled_1d()
    with pytest.raises(TypeError):
        h - h


def test_pick_each_raises():
    h = bh.Histogram(
        bh.axis.Regular(2, 0, 2),
        bh.axis.Regular(2, 0, 2),
        storage=bh.storage.Collector(),
    )
    with pytest.raises(NotImplementedError):
        h[0, :]


def test_group_rebin_raises():
    h = bh.Histogram(bh.axis.Regular(4, 0, 4), storage=bh.storage.Collector())
    with pytest.raises(NotImplementedError):
        h[:: bh.rebin(groups=[1, 3])]


def test_threaded_fill_raises():
    h = bh.Histogram(bh.axis.Regular(2, 0, 2), storage=bh.storage.Collector())
    with pytest.raises(RuntimeError):
        h.fill([0, 1], sample=[1.0, 2.0], threads=2)


def test_weighted_fill_raises():
    h = bh.Histogram(bh.axis.Regular(2, 0, 2), storage=bh.storage.Collector())
    with pytest.raises(ValueError):
        h.fill([0, 1], sample=[1.0, 2.0], weight=[1.0, 1.0])


def test_structural_match():
    h = bh.Histogram(bh.axis.Regular(3, 0, 3), storage=bh.storage.Collector())
    match h:
        case bh.Histogram(storage=bh.storage.Collector()):
            pass
        case _:
            raise AssertionError("Collector storage did not match")


# --- The Values cell accumulator ---


def test_values_construction():
    c = V([1.0, 2.0, 3.0])
    assert len(c) == 3
    assert c[0] == 1.0
    assert c[-1] == 3.0
    assert list(c) == [1.0, 2.0, 3.0]
    assert_array_equal(c.value, [1.0, 2.0, 3.0])
    assert len(V()) == 0


def test_values_index_error():
    c = V([1.0])
    with pytest.raises(IndexError):
        c[5]


def test_values_repr():
    assert repr(V([1.0, 2.0])) == "Values([1.0, 2.0])"


def test_values_equality_copy_pickle():
    c = V([1.0, 2.0])
    assert c == V([1.0, 2.0])
    assert c != V([1.0])
    assert c != 1.0
    assert copy.copy(c) == c
    assert copy.deepcopy(c) == c
    assert pickle.loads(pickle.dumps(c)) == c


def test_values_in_accumulators():
    assert V is bh.accumulators.Values
    assert V.__module__ == "boost_histogram.accumulators"
