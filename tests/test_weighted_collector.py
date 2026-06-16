from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import boost_histogram as bh

WV = bh.accumulators.WeightedValues
ENTRY_DTYPE = np.dtype([("value", "<f8"), ("weight", "<f8")])


def _filled_1d():
    """A 1D WeightedCollector histogram holding (value, weight) pairs per bin."""
    h = bh.Histogram(bh.axis.Regular(3, 0, 3), storage=bh.storage.WeightedCollector())
    h.fill(
        [0, 0, 1, 2, 2, 2],
        sample=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        weight=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    return h


def test_construction_and_repr():
    h = bh.Histogram(bh.axis.Regular(3, 0, 3), storage=bh.storage.WeightedCollector())
    assert isinstance(h.storage_type(), bh.storage.WeightedCollector)
    assert repr(bh.storage.WeightedCollector()) == "WeightedCollector()"
    assert "WeightedCollector()" in repr(h)
    # str must not raise for a 1D histogram (no scalar rendering for ragged cells)
    assert "WeightedCollector" in str(h)


def test_fill_and_index():
    h = _filled_1d()
    # h[i] is a WeightedValues accumulator (the per-bin cell)
    assert isinstance(h[0], WV)
    assert list(h[0]) == [(1.0, 0.1), (2.0, 0.2)]
    assert list(h[1]) == [(3.0, 0.3)]
    assert list(h[2]) == [(4.0, 0.4), (5.0, 0.5), (6.0, 0.6)]
    # individual entries are (value, weight) tuples
    assert h[0][0] == (1.0, 0.1)
    assert len(h[0]) == 2


def test_value_and_weight_columns():
    h = _filled_1d()
    assert_array_equal(h[0].value, [1.0, 2.0])
    assert_array_equal(h[0].weight, [0.1, 0.2])
    assert h[0].value.dtype == np.float64
    assert_array_equal(h[2].value, [4.0, 5.0, 6.0])
    assert_array_equal(h[2].weight, [0.4, 0.5, 0.6])


def test_weight_omitted_defaults_to_one():
    h = bh.Histogram(bh.axis.Regular(2, 0, 2), storage=bh.storage.WeightedCollector())
    h.fill([0, 1], sample=[1.0, 2.0])
    assert list(h[0]) == [(1.0, 1.0)]
    assert list(h[1]) == [(2.0, 1.0)]


def test_scalar_weight_broadcast():
    h = bh.Histogram(bh.axis.Regular(2, 0, 2), storage=bh.storage.WeightedCollector())
    h.fill([0, 1, 1], sample=[1.0, 2.0, 3.0], weight=0.5)
    assert list(h[0]) == [(1.0, 0.5)]
    assert list(h[1]) == [(2.0, 0.5), (3.0, 0.5)]


def test_sample_required():
    # Same error as WeightedMean and Collector when the sample is missing
    h = bh.Histogram(bh.axis.Regular(2, 0, 2), storage=bh.storage.WeightedCollector())
    with pytest.raises(ValueError, match="Sample array must be 1D"):
        h.fill([0, 1])


def test_view_structured_cells():
    h = _filled_1d()
    v = h.view()
    assert v.dtype == object
    assert v.shape == (3,)
    # each cell is its own structured float64 array
    assert v[0].dtype == ENTRY_DTYPE
    assert_array_equal(v[0]["value"], [1.0, 2.0])
    assert_array_equal(v[0]["weight"], [0.1, 0.2])
    assert_array_equal(v[2]["value"], [4.0, 5.0, 6.0])
    assert_array_equal(v[2]["weight"], [0.4, 0.5, 0.6])


def test_view_flow():
    h = bh.Histogram(bh.axis.Regular(2, 0, 2), storage=bh.storage.WeightedCollector())
    h.fill([-1, 0, 1, 5], sample=[10.0, 1.0, 2.0, 99.0], weight=[1.0, 2.0, 3.0, 4.0])
    assert [c.tolist() for c in h.view()] == [[(1.0, 2.0)], [(2.0, 3.0)]]
    assert [c.tolist() for c in h.view(flow=True)] == [
        [(10.0, 1.0)],
        [(1.0, 2.0)],
        [(2.0, 3.0)],
        [(99.0, 4.0)],
    ]


def test_scalar_arg_broadcast():
    h = bh.Histogram(bh.axis.Regular(1, 0, 1), storage=bh.storage.WeightedCollector())
    h.fill(0, sample=[1.0, 2.0, 3.0], weight=[0.1, 0.2, 0.3])
    assert list(h[0]) == [(1.0, 0.1), (2.0, 0.2), (3.0, 0.3)]


def test_2d():
    h = bh.Histogram(
        bh.axis.Regular(2, 0, 2),
        bh.axis.Regular(2, 0, 2),
        storage=bh.storage.WeightedCollector(),
    )
    h.fill([0, 0, 1], [0, 1, 1], sample=[1.0, 2.0, 3.0], weight=[0.1, 0.2, 0.3])
    v = h.view()
    assert v.shape == (2, 2)
    assert list(h[0, 0]) == [(1.0, 0.1)]
    assert list(h[0, 1]) == [(2.0, 0.2)]
    assert list(h[1, 0]) == []
    assert list(h[1, 1]) == [(3.0, 0.3)]


def test_sum_is_concatenation():
    h = _filled_1d()
    assert isinstance(h.sum(), WV)
    assert sorted(h.sum()) == [
        (1.0, 0.1),
        (2.0, 0.2),
        (3.0, 0.3),
        (4.0, 0.4),
        (5.0, 0.5),
        (6.0, 0.6),
    ]
    # empty histogram sums to an empty cell
    empty = bh.Histogram(
        bh.axis.Regular(3, 0, 3), storage=bh.storage.WeightedCollector()
    )
    assert list(empty.sum()) == []


def test_addition_concatenates():
    h = _filled_1d()
    hh = h + h
    assert list(hh[0]) == [(1.0, 0.1), (2.0, 0.2), (1.0, 0.1), (2.0, 0.2)]
    assert list(hh[1]) == [(3.0, 0.3), (3.0, 0.3)]


def test_project_concatenates():
    h = bh.Histogram(
        bh.axis.Regular(2, 0, 2),
        bh.axis.Regular(2, 0, 2),
        storage=bh.storage.WeightedCollector(),
    )
    h.fill([0, 0, 1], [0, 1, 1], sample=[1.0, 2.0, 3.0], weight=[0.1, 0.2, 0.3])
    p = h.project(0)
    assert list(p[0]) == [(1.0, 0.1), (2.0, 0.2)]
    assert list(p[1]) == [(3.0, 0.3)]


def test_slice_reduce():
    h = bh.Histogram(bh.axis.Regular(4, 0, 4), storage=bh.storage.WeightedCollector())
    h.fill([0, 1, 2, 3], sample=[1.0, 2.0, 3.0, 4.0], weight=[0.1, 0.2, 0.3, 0.4])
    sub = h[1:3]
    assert sub.ndim == 1
    assert [c.tolist() for c in sub.view()] == [[(2.0, 0.2)], [(3.0, 0.3)]]


def test_factor_rebin_concatenates():
    h = bh.Histogram(bh.axis.Regular(4, 0, 4), storage=bh.storage.WeightedCollector())
    h.fill([0, 1, 2, 3], sample=[1.0, 2.0, 3.0, 4.0], weight=[0.1, 0.2, 0.3, 0.4])
    rebinned = h[:: bh.rebin(2)]
    assert [c.tolist() for c in rebinned.view()] == [
        [(1.0, 0.1), (2.0, 0.2)],
        [(3.0, 0.3), (4.0, 0.4)],
    ]


def test_pickle_roundtrip():
    h = _filled_1d()
    h2 = pickle.loads(pickle.dumps(h))
    assert h2 == h
    assert list(h2[2]) == [(4.0, 0.4), (5.0, 0.5), (6.0, 0.6)]


@pytest.mark.parametrize(
    "corrupt",
    [
        [-1, 2, 1, 3, 0],  # negative count
        [0, 2, 1, 4, 0],  # counts sum exceeds value buffer
        [0, 2, 1, 2, 0],  # counts sum falls short of value buffer
    ],
)
def test_pickle_rejects_corrupt_counts(corrupt):
    # Counts that don't describe the value/weight buffers exactly must be rejected
    # before native code reads past the NumPy buffers (see PR #1133 review).
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
    assert [c.tolist() for c in h.view()] == [[], [], []]


def test_equality():
    assert _filled_1d() == _filled_1d()
    other = bh.Histogram(
        bh.axis.Regular(3, 0, 3), storage=bh.storage.WeightedCollector()
    )
    other.fill([0], sample=[1.0], weight=[0.1])
    assert _filled_1d() != other
    # same values, different weights
    different_weights = bh.Histogram(
        bh.axis.Regular(3, 0, 3), storage=bh.storage.WeightedCollector()
    )
    different_weights.fill(
        [0, 0, 1, 2, 2, 2], sample=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], weight=1.0
    )
    assert _filled_1d() != different_weights


def test_not_equal_to_plain_collector():
    weighted = bh.Histogram(
        bh.axis.Regular(3, 0, 3), storage=bh.storage.WeightedCollector()
    )
    plain = bh.Histogram(bh.axis.Regular(3, 0, 3), storage=bh.storage.Collector())
    weighted.fill([0], sample=[1.0])
    plain.fill([0], sample=[1.0])
    assert weighted != plain


# --- Unsupported operations (the object view is copy-only) ---


def test_setitem_raises():
    h = _filled_1d()
    with pytest.raises(NotImplementedError):
        h[0] = [(1.0, 2.0)]


def test_scalar_arithmetic_raises():
    h = _filled_1d()
    with pytest.raises(NotImplementedError, match="WeightedCollector"):
        h * 2


def test_histogram_subtraction_raises():
    h = _filled_1d()
    with pytest.raises(TypeError):
        h - h


def test_pick_each_raises():
    h = bh.Histogram(
        bh.axis.Regular(2, 0, 2),
        bh.axis.Regular(2, 0, 2),
        storage=bh.storage.WeightedCollector(),
    )
    with pytest.raises(NotImplementedError):
        h[0, :]


def test_group_rebin_raises():
    h = bh.Histogram(bh.axis.Regular(4, 0, 4), storage=bh.storage.WeightedCollector())
    with pytest.raises(NotImplementedError):
        h[:: bh.rebin(groups=[1, 3])]


def test_threaded_fill_raises():
    h = bh.Histogram(bh.axis.Regular(2, 0, 2), storage=bh.storage.WeightedCollector())
    with pytest.raises(RuntimeError):
        h.fill([0, 1], sample=[1.0, 2.0], threads=2)


def test_structural_match():
    h = bh.Histogram(bh.axis.Regular(3, 0, 3), storage=bh.storage.WeightedCollector())
    match h:
        case bh.Histogram(storage=bh.storage.WeightedCollector()):
            pass
        case _:
            raise AssertionError("WeightedCollector storage did not match")


# --- The WeightedValues cell accumulator ---


def test_weighted_values_construction():
    c = WV([(1.0, 0.1), (2.0, 0.2)])
    assert len(c) == 2
    assert c[0] == (1.0, 0.1)
    assert c[-1] == (2.0, 0.2)
    assert list(c) == [(1.0, 0.1), (2.0, 0.2)]
    assert_array_equal(c.value, [1.0, 2.0])
    assert_array_equal(c.weight, [0.1, 0.2])
    assert len(WV()) == 0


def test_weighted_values_index_error():
    c = WV([(1.0, 0.1)])
    with pytest.raises(IndexError):
        c[5]


def test_weighted_values_repr():
    assert repr(WV([(1.0, 0.5)])) == "WeightedValues([(1.0, 0.5)])"


def test_weighted_values_equality_copy_pickle():
    c = WV([(1.0, 0.1), (2.0, 0.2)])
    assert c == WV([(1.0, 0.1), (2.0, 0.2)])
    assert c != WV([(1.0, 0.1)])
    assert c != (1.0, 0.1)
    assert copy.copy(c) == c
    assert copy.deepcopy(c) == c
    assert pickle.loads(pickle.dumps(c)) == c


def test_weighted_values_in_accumulators():
    assert WV is bh.accumulators.WeightedValues
    assert WV.__module__ == "boost_histogram.accumulators"
