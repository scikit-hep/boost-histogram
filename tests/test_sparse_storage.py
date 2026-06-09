from __future__ import annotations

import copy
import json
import operator
import warnings
from pickle import dumps, loads

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import boost_histogram as bh
from boost_histogram.serialization import from_uhi, to_uhi


def coo_dict(h):
    """Map of (multi-index) -> value for the filled cells, order-independent."""
    indices, values = h.to_coo()
    return {tuple(int(i[j]) for i in indices): v for j, v in enumerate(values)}


def pickle_roundtrip(x):
    return loads(dumps(x))


def test_construct_and_fill():
    h = bh.Histogram(bh.axis.Regular(1000, 0, 1000), storage=bh.storage.DoubleSparse())
    assert h.storage_type is bh.storage.DoubleSparse

    h.fill([1.5, 1.5, 5.5, 900.5])
    assert h.sum() == 4.0
    # Point reads go through C++ at(), not the (unavailable) dense view.
    assert h[1] == 2.0
    assert h[5] == 1.0
    assert h[900] == 1.0
    assert h[2] == 0.0


def test_stays_sparse_after_ops():
    h = bh.Histogram(bh.axis.Regular(100, 0, 100), storage=bh.storage.DoubleSparse())
    h.fill([1.5, 50.5, 50.5])

    # Slicing and projection use C++ algorithms and must keep sparse storage.
    sliced = h[10:60]
    assert sliced.storage_type is bh.storage.DoubleSparse
    assert sliced.sum() == 2.0

    h2 = bh.Histogram(
        bh.axis.Regular(5, 0, 5),
        bh.axis.Regular(5, 0, 5),
        storage=bh.storage.DoubleSparse(),
    )
    h2.fill([1.5, 1.5], [2.5, 2.5])
    projected = h2.project(0)
    assert projected.storage_type is bh.storage.DoubleSparse
    assert projected.sum() == 2.0


def test_to_coo_filled_cells_only():
    h = bh.Histogram(
        bh.axis.Regular(5, 0, 5),
        bh.axis.IntCategory([10, 20, 30]),
        storage=bh.storage.DoubleSparse(),
    )
    h.fill([1.5, 1.5, 3.5], [10, 10, 30])

    indices, values = h.to_coo()
    assert isinstance(indices, tuple)
    assert len(indices) == 2
    # One entry per distinct filled bin.
    assert len(values) == 2
    assert coo_dict(h) == {(1, 0): 2.0, (3, 2): 1.0}


def test_to_coo_flow():
    h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.DoubleSparse())
    h.fill([-5, 5.5, 100])  # underflow, normal bin 5, overflow

    # Without flow, the flow cells are dropped, indices run 0..size-1.
    assert coo_dict(h) == {(5,): 1.0}

    # With flow, underflow sits at index 0, overflow at extent-1 (== 11 here).
    indices_f, values_f = h.to_coo(flow=True)
    flow_map = {int(indices_f[0][j]): v for j, v in enumerate(values_f)}
    assert flow_map == {0: 1.0, 6: 1.0, 11: 1.0}


def test_view_and_asarray_raise_but_copies_densify():
    h = bh.Histogram(bh.axis.Regular(5, 0, 5), storage=bh.storage.DoubleSparse())
    h.fill([1.5])

    # The zero-copy buffer view (and implicit array conversion) are unsupported.
    with pytest.raises(TypeError, match="to_coo"):
        h.view()
    with pytest.raises(TypeError):
        np.asarray(h)

    # The explicit copying accessors densify and keep working.
    assert_array_equal(h.values(), [0, 1, 0, 0, 0])
    assert_array_equal(h.counts(), [0, 1, 0, 0, 0])
    assert_array_equal(h.variances(), [0, 1, 0, 0, 0])


def test_copy_accessors_match_dense():
    axes = (bh.axis.Regular(10, 0, 10), bh.axis.IntCategory([5, 6, 7]))
    x = np.array([-1, 0.5, 0.5, 3.5, 9.5, 100])
    y = np.array([5, 6, 6, 7, 5, 6])

    dense = bh.Histogram(*axes, storage=bh.storage.Double())
    dense.fill(x, y)
    sparse = bh.Histogram(*axes, storage=bh.storage.DoubleSparse())
    sparse.fill(x, y)

    for flow in (False, True):
        assert_array_equal(sparse.values(flow=flow), dense.values(flow=flow))
        assert_array_equal(sparse.counts(flow=flow), dense.counts(flow=flow))
        assert_array_equal(sparse.variances(flow=flow), dense.variances(flow=flow))


def test_to_coo_requires_sparse():
    h = bh.Histogram(bh.axis.Regular(5, 0, 5), storage=bh.storage.Double())
    with pytest.raises(TypeError, match="sparse"):
        h.to_coo()


def test_repr_and_str_do_not_densify():
    # A huge axis would be impossible to densify; repr/str must still work.
    h = bh.Histogram(bh.axis.Regular(10**9, 0, 1), storage=bh.storage.DoubleSparse())
    h.fill([0.5, 0.5])
    assert "DoubleSparse" in repr(h)
    assert "Sum: 2" in repr(h)
    # __str__ falls back to repr for sparse rather than rendering every bin.
    assert "DoubleSparse" in str(h)


@pytest.mark.parametrize("copy_fn", [copy.copy, copy.deepcopy, pickle_roundtrip])
def test_copy_pickle_roundtrip(copy_fn):
    h = bh.Histogram(
        bh.axis.Regular(20, 0, 20),
        bh.axis.IntCategory([1, 2, 3]),
        storage=bh.storage.DoubleSparse(),
    )
    h.fill([-1, 5.5, 5.5, 100], [1, 2, 2, 3])

    h2 = copy_fn(h)
    assert h2 == h
    assert h2.sum(flow=True) == h.sum(flow=True)
    assert coo_dict(h2) == coo_dict(h)


def test_uhi_roundtrip_with_data():
    h = bh.Histogram(
        bh.axis.Regular(10, 0, 10),
        bh.axis.Regular(8, 0, 8),
        storage=bh.storage.DoubleSparse(),
    )
    h.fill([-1, 2.5, 2.5, 100], [3, 4, 4, 200])

    data = to_uhi(h)
    assert data["storage"]["type"] == "double"
    assert data["storage"]["sparse"] is True
    assert data["writer_info"]["boost-histogram"]["storage_type"] == "DoubleSparse"

    h2 = from_uhi(data)
    assert h2.storage_type is bh.storage.DoubleSparse
    assert h2 == h
    assert h2.sum(flow=True) == h.sum(flow=True)


def test_uhi_roundtrip_through_json():
    h = bh.Histogram(
        bh.axis.Regular(10, 0, 10),
        bh.axis.Regular(6, 0, 6),
        storage=bh.storage.DoubleSparse(),
    )
    h.fill([2.5, 2.5, 8.5], [1.5, 1.5, 3.5])

    def to_jsonable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_jsonable(v) for v in obj]
        return obj

    data = json.loads(json.dumps(to_jsonable(to_uhi(h))))
    h2 = from_uhi(data)
    assert h2 == h


def test_uhi_structure_only():
    h = bh.Histogram(bh.axis.Regular(5, 0, 5), storage=bh.storage.DoubleSparse())
    h.fill([1.5, 1.5])

    data = to_uhi(h, keep_storage=False)
    h2 = from_uhi(data)
    assert h2.storage_type is bh.storage.DoubleSparse
    assert h2.sum(flow=True) == 0.0


def test_uhi_roundtrip_empty():
    h = bh.Histogram(bh.axis.Regular(5, 0, 5), storage=bh.storage.DoubleSparse())
    h2 = from_uhi(to_uhi(h))
    assert h2 == h
    assert h2.sum() == 0.0


def test_setitem_raises_on_sparse():
    """Sparse storage does not support direct assignment (no dense buffer)."""
    h = bh.Histogram(bh.axis.Regular(5, 0, 5), storage=bh.storage.DoubleSparse())
    with pytest.raises(TypeError, match="buffer"):
        h[0] = 1.0


def test_sparse_histogram_addition():
    """Adding two sparse histograms stays sparse and combines counts correctly."""
    h1 = bh.Histogram(bh.axis.Regular(5, 0, 5), storage=bh.storage.DoubleSparse())
    h1.fill([1.5, 3.5])

    h2 = bh.Histogram(bh.axis.Regular(5, 0, 5), storage=bh.storage.DoubleSparse())
    h2.fill([1.5, 4.5])

    h_sum = h1 + h2
    assert h_sum.storage_type is bh.storage.DoubleSparse
    assert h_sum[1] == 2.0
    assert h_sum[3] == 1.0
    assert h_sum[4] == 1.0
    assert h_sum[2] == 0.0

    # densified view should match a dense histogram with the same fills
    dense = bh.Histogram(bh.axis.Regular(5, 0, 5), storage=bh.storage.Double())
    dense.fill([1.5, 1.5, 3.5, 4.5])
    assert_array_equal(h_sum.values(), dense.values())


@pytest.mark.parametrize("op", ["mul", "truediv"])
def test_scalar_mul_div_stays_sparse(op):
    """Scalar * and / scale filled cells in place and preserve sparsity."""
    fn = getattr(operator, op)
    sparse = bh.Histogram(bh.axis.Regular(5, 0, 5), storage=bh.storage.DoubleSparse())
    sparse.fill([-1, 1.5, 1.5, 3.5, 100])  # includes flow cells
    dense = bh.Histogram(bh.axis.Regular(5, 0, 5), storage=bh.storage.Double())
    dense.fill([-1, 1.5, 1.5, 3.5, 100])

    result = fn(sparse, 2.0)
    assert result.storage_type is bh.storage.DoubleSparse
    # Original is untouched (operator returns a copy).
    assert_array_equal(sparse.values(flow=True), dense.values(flow=True))
    for flow in (False, True):
        assert_array_equal(result.values(flow=flow), fn(dense, 2.0).values(flow=flow))


def test_scalar_imul_in_place_stays_sparse():
    h = bh.Histogram(bh.axis.Regular(5, 0, 5), storage=bh.storage.DoubleSparse())
    h.fill([1.5, 1.5, 3.5])
    h *= 3
    assert h.storage_type is bh.storage.DoubleSparse
    assert_array_equal(h.values(), [0, 6, 0, 3, 0])


@pytest.mark.parametrize("op", ["+", "-"])
def test_scalar_add_sub_refused(op):
    """Scalar +/- would fill every cell, so sparse storage refuses them."""
    fn = {"+": operator.add, "-": operator.sub}[op]
    h = bh.Histogram(bh.axis.Regular(5, 0, 5), storage=bh.storage.DoubleSparse())
    h.fill([1.5])
    with pytest.raises(TypeError, match="fill every cell"):
        fn(h, 2.0)


def _mixed_axes_hist():
    h = bh.Histogram(
        bh.axis.Regular(5, 0, 5),
        bh.axis.IntCategory([10, 20, 30]),
        storage=bh.storage.DoubleSparse(),
    )
    h.fill([1.5, 1.5, 3.5], [10, 10, 30])
    return h


def test_mixed_int_slice_indexing_raises():
    """Mixed integer + slice indexing routes through .view() and raises."""
    h = _mixed_axes_hist()
    with pytest.raises(TypeError, match=r"to_coo|sparse"):
        _ = h[2, :]


def test_list_pick_indexing_raises():
    """Categorical list selection routes through .view() and raises."""
    h = _mixed_axes_hist()
    # The list-pick path is experimental and warns before it reaches the view.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(TypeError, match=r"to_coo|sparse"):
            _ = h[:, [0, 2]]


def test_vectorized_array_indexing_raises():
    h = bh.Histogram(bh.axis.Regular(5, 0, 5), storage=bh.storage.DoubleSparse())
    h.fill([1.5, 3.5])
    with pytest.raises(TypeError, match=r"to_coo|sparse"):
        _ = h[np.array([0, 1, 2])]


@pytest.mark.parametrize(
    "make_index",
    [
        pytest.param(lambda: slice(1, 4), id="slice"),
        pytest.param(lambda: slice(None, None, bh.rebin(2)), id="rebin"),
        pytest.param(lambda: slice(None, None, sum), id="sum"),
    ],
)
def test_reduce_indexing_stays_sparse(make_index):
    """Slicing, rebin, and integration delegate to C++ and stay sparse."""
    h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.DoubleSparse())
    h.fill([1.5, 5.5, 5.5, 8.5])
    result = h[make_index()]
    if isinstance(result, bh.Histogram):
        assert result.storage_type is bh.storage.DoubleSparse


def test_matches_dense_values():
    # The filled cells should agree with what a dense Double histogram records.
    axes = (bh.axis.Regular(10, 0, 10), bh.axis.IntCategory([5, 6, 7]))
    x = np.array([0.5, 0.5, 3.5, 9.5])
    y = np.array([5, 6, 6, 7])

    dense = bh.Histogram(*axes, storage=bh.storage.Double())
    dense.fill(x, y)
    sparse = bh.Histogram(*axes, storage=bh.storage.DoubleSparse())
    sparse.fill(x, y)

    dense_view = dense.view()
    rebuilt = np.zeros_like(dense_view)
    indices, values = sparse.to_coo()
    rebuilt[indices] = values
    assert_array_equal(rebuilt, dense_view)
