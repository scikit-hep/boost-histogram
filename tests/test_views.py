from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

import boost_histogram as bh


@pytest.fixture
def v():
    h = bh.Histogram(bh.axis.Integer(0, 4), storage=bh.storage.Weight())
    h.fill([1, 1, 1, 2, 2, 3])
    return h.view()


def test_basic_view(v):
    assert v.value == approx([0, 3, 2, 1])
    assert v.variance == approx([0, 3, 2, 1])


def test_view_mul(v):
    v2 = v * 2
    assert v2.value == approx([0, 6, 4, 2])
    assert v2.variance == approx([0, 12, 8, 4])

    v2 = 2 * v
    assert v2.value == approx([0, 6, 4, 2])
    assert v2.variance == approx([0, 12, 8, 4])

    v2 = v * (-2)
    assert v2.value == approx([0, -6, -4, -2])
    assert v2.variance == approx([0, 12, 8, 4])

    v *= 2
    assert v.value == approx([0, 6, 4, 2])
    assert v.variance == approx([0, 12, 8, 4])


def test_view_div(v):
    v2 = v / 2
    assert v2.value == approx([0, 1.5, 1, 0.5])
    assert v2.variance == approx([0, 0.75, 0.5, 0.25])

    v2 = v / (-0.5)
    assert v2.value == approx([0, -6, -4, -2])
    assert v2.variance == approx([0, 12, 8, 4])

    # Issue #1143 (B11): the reciprocal of a weighted sum has no meaningful
    # variance, so dividing by a view is rejected.
    with pytest.raises(TypeError, match="divide a scalar or array"):
        1 / v[1:]

    v /= 0.5
    assert v.value == approx([0, 6, 4, 2])
    assert v.variance == approx([0, 12, 8, 4])


def test_view_add(v):
    v2 = v + 1
    assert v2.value == approx([1, 4, 3, 2])
    assert v2.variance == approx([1, 4, 3, 2])

    v2 = v + 2
    assert v2.value == approx([2, 5, 4, 3])
    assert v2.variance == approx([4, 7, 6, 5])

    v2 = 2 + v
    assert v2.value == approx([2, 5, 4, 3])
    assert v2.variance == approx([4, 7, 6, 5])

    v2 = v.copy()
    v2 += 2
    assert v2.value == approx([2, 5, 4, 3])
    assert v2.variance == approx([4, 7, 6, 5])

    v2 = v + v
    assert v2.value == approx(v.value * 2)
    assert v2.variance == approx(v.variance * 2)


def test_view_sub(v):
    v2 = v - 1
    assert v2.value == approx([-1, 2, 1, 0])
    assert v2.variance == approx([1, 4, 3, 2])

    v2 = v - 2
    assert v2.value == approx([-2, 1, 0, -1])
    assert v2.variance == approx([4, 7, 6, 5])

    v2 = 1 - v
    assert v2.value == approx([1, -2, -1, 0])
    assert v2.variance == approx([1, 4, 3, 2])

    v2 = v.copy()
    v2 -= 2
    assert v2.value == approx([-2, 1, 0, -1])
    assert v2.variance == approx([4, 7, 6, 5])

    v2 = v - v
    assert v2.value == approx([0, 0, 0, 0])
    assert v2.variance == approx(v.variance * 2)


def test_view_unary(v):
    v2 = +v
    assert v.value == approx(v2.value)
    assert v.variance == approx(v2.variance)

    v2 = -v
    assert -v.value == approx(v2.value)
    assert v.variance == approx(v2.variance)


def test_view_add_same(v):
    v2 = v + v

    assert v.value * 2 == approx(v2.value)
    assert v.variance * 2 == approx(v2.variance)

    v2 = v + v[1]
    assert v.value + 3 == approx(v2.value)
    assert v.variance + 3 == approx(v2.variance)

    v2 = v + bh.accumulators.WeightedSum(5, 6)
    assert v.value + 5 == approx(v2.value)
    assert v.variance + 6 == approx(v2.variance)

    with pytest.raises(TypeError):
        v2 = v + bh.accumulators.WeightedMean(1, 2, 5, 6)


def test_view_assign(v):
    v[...] = [[4, 1], [5, 2], [6, 1], [7, 2]]

    assert v.value == approx([4, 5, 6, 7])
    assert v.variance == approx([1, 2, 1, 2])


def test_view_assign_mean():
    h = bh.Histogram(bh.axis.Integer(0, 4), storage=bh.storage.Mean())
    m = h.copy().view()

    h[...] = [[10, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    assert h.view().count == approx([10, 4, 7, 10])
    assert h.view().value == approx([2, 5, 8, 11])
    assert h.view().variance == approx([3, 6, 9, 12])

    # Make sure this really was a copy
    assert m.count[0] != 10

    # Assign directly on view
    m[...] = [[10, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

    assert m.count == approx([10, 4, 7, 10])
    assert m.value == approx([2, 5, 8, 11])
    assert m.variance == approx([3, 6, 9, 12])
    # Note: if counts <= 1, variance is undefined


def test_view_assign_wmean():
    h = bh.Histogram(bh.axis.Integer(0, 4), storage=bh.storage.WeightedMean())

    w = h.copy().view()

    h[...] = [[10, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]

    assert h.view().sum_of_weights == approx([10, 5, 9, 13])
    assert h.view().sum_of_weights_squared == approx([2, 6, 10, 14])
    assert h.view().value == approx([3, 7, 11, 15])
    assert h.view().variance == approx([4, 8, 12, 16])

    # Make sure this really was a copy
    assert w.sum_of_weights[0] != 10

    # Assign directly on view
    w[...] = [[10, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]

    assert w.sum_of_weights == approx([10, 5, 9, 13])
    assert w.sum_of_weights_squared == approx([2, 6, 10, 14])
    assert w.value == approx([3, 7, 11, 15])
    assert w.variance == approx([4, 8, 12, 16])
    # Note: if sum_of_weights <= 1, variance is undefined

    w[0] = [9, 1, 2, 3]
    assert w.sum_of_weights[0] == 9
    assert w[0].sum_of_weights_squared == 1
    assert w.value[0] == 2
    assert w[0].variance == 3


# Issue #826 - accessing fields of a 0-d MeanView raised "iteration over a 0-d array"
def test_0d_mean_view():
    h = bh.Histogram(bh.axis.Integer(0, 4), storage=bh.storage.Mean())
    h.fill([0, 1, 2, 3, 0, 1], sample=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    profile = h.project()
    assert profile.shape == ()

    # view() should return a 0-d MeanView without error
    v = profile.view()
    assert v.shape == ()

    # Field access on a 0-d MeanView should work
    assert v.count == approx(6)
    assert v.value == approx((1 + 2 + 3 + 4 + 5 + 6) / 6)
    assert v["value"] == approx((1 + 2 + 3 + 4 + 5 + 6) / 6)

    # values(), variances(), and counts() on the histogram should work
    assert profile.values() == approx((1 + 2 + 3 + 4 + 5 + 6) / 6)
    assert profile.counts() == approx(6)
    assert profile.variances() == approx(v.variance / 6)


def test_0d_weighted_mean_view():
    h = bh.Histogram(bh.axis.Integer(0, 4), storage=bh.storage.WeightedMean())
    h.fill([0, 1, 2, 3], sample=[1.0, 2.0, 3.0, 4.0], weight=[1.0, 1.0, 1.0, 1.0])
    profile = h.project()
    assert profile.shape == ()

    # view() should return a 0-d WeightedMeanView without error
    v = profile.view()
    assert v.shape == ()

    # Field access on a 0-d WeightedMeanView should work
    assert v.value == approx(2.5)
    assert v.sum_of_weights == approx(4.0)
    assert v["value"] == approx(2.5)

    # values() and counts() on the histogram should work
    assert profile.values() == approx(2.5)
    assert profile.counts() == approx(4.0)


@pytest.fixture
def v2d():
    h = bh.Histogram(
        bh.axis.Integer(0, 2), bh.axis.Integer(0, 2), storage=bh.storage.Weight()
    )
    h.fill([0, 0, 1, 1], [0, 1, 0, 1], weight=[1, 2, 1, 1])
    return h.view()


# Issue #1143 (B11): ``scalar / view`` would produce a statistically
# meaningless variance (s**2 / variance), so it must raise like the mean views.
def test_view_rdiv_rejected(v):
    with pytest.raises(TypeError, match="divide a scalar or array"):
        2.0 / v
    with pytest.raises(TypeError, match="divide a scalar or array"):
        2 // v
    with pytest.raises(TypeError, match="divide a scalar or array"):
        np.true_divide(2.0, v)
    with pytest.raises(TypeError, match="divide a scalar or array"):
        np.ones(4) / v

    # Scaling in the other direction must keep working.
    assert (v / 2).value == approx(v.value / 2)
    assert (2 * v).value == approx(v.value * 2)


# Issue #1143 (B2): a caller-supplied ``out=`` used to be forwarded into the
# per-field reductions, so every field overwrote the same buffer.
def test_view_reduce_out_mismatched_dtype(v2d):
    with pytest.raises(TypeError, match="out="):
        np.sum(v2d, axis=0, out=np.empty(2))


def test_view_reduce_out_matching_dtype(v2d):
    out = np.zeros(2, dtype=v2d.dtype)
    result = np.sum(v2d, axis=0, out=out)

    assert np.shares_memory(result, out)
    assert isinstance(result, type(v2d))
    assert out["value"] == approx(np.sum(v2d["value"], axis=0))
    assert out["variance"] == approx(np.sum(v2d["variance"], axis=0))


def test_view_reduce_out_full(v2d):
    out = np.zeros((), dtype=v2d.dtype)
    result = np.sum(v2d, out=out)

    assert np.shares_memory(result, out)
    assert out["value"] == approx(np.sum(v2d["value"]))
    assert out["variance"] == approx(np.sum(v2d["variance"]))


def test_view_reduce_axis(v2d):
    result = np.sum(v2d, axis=0)
    assert result["value"] == approx([2, 3])
    assert result["variance"] == approx([2, 5])


# Issue #1143 (B12): binary ops used to preallocate the result with the view's
# own shape, breaking legal broadcasts.
def test_view_broadcast_add(v):
    arr = np.ones((2, 4))
    for result in (v + arr, arr + v):
        assert result.shape == (2, 4)
        assert isinstance(result, type(v))
        assert result.value == approx(np.broadcast_to(v.value + 1, (2, 4)))
        assert result.variance == approx(np.broadcast_to(v.variance + 1, (2, 4)))


def test_view_broadcast_mul(v):
    arr = np.full((2, 4), 2.0)
    for result in (v * arr, arr * v):
        assert result.shape == (2, 4)
        assert isinstance(result, type(v))
        assert result.value == approx(np.broadcast_to(v.value * 2, (2, 4)))
        assert result.variance == approx(np.broadcast_to(v.variance * 4, (2, 4)))


@pytest.fixture
def mean_pair():
    # Overlapping fills, each with at least one empty bin, to exercise the
    # zero-count merge guard.
    h1 = bh.Histogram(bh.axis.Integer(0, 4), storage=bh.storage.Mean())
    h1.fill([0, 0, 1, 3], sample=[1.0, 2.0, 3.0, 9.0])  # bin 2 empty
    h2 = bh.Histogram(bh.axis.Integer(0, 4), storage=bh.storage.Mean())
    h2.fill([0, 2, 2], sample=[5.0, 6.0, 8.0])  # bins 1 and 3 empty
    return h1, h2


@pytest.fixture
def weighted_mean_pair():
    h1 = bh.Histogram(bh.axis.Integer(0, 4), storage=bh.storage.WeightedMean())
    h1.fill([0, 0, 1, 3], sample=[1.0, 2.0, 3.0, 9.0], weight=[1.0, 2.0, 1.0, 3.0])
    h2 = bh.Histogram(bh.axis.Integer(0, 4), storage=bh.storage.WeightedMean())
    h2.fill([0, 2, 2], sample=[5.0, 6.0, 8.0], weight=[2.0, 1.0, 1.0])
    return h1, h2


def test_mean_view_combine_matches_hist(mean_pair):
    h1, h2 = mean_pair
    combined = (h1 + h2).view()
    summed = h1.view() + h2.view()

    assert isinstance(summed, type(h1.view()))
    for field in combined.dtype.names:
        assert summed[field] == approx(combined[field])
    assert summed.value == approx(combined.value)
    assert summed.variance == approx(combined.variance, nan_ok=True)


def test_weighted_mean_view_combine_matches_hist(weighted_mean_pair):
    h1, h2 = weighted_mean_pair
    combined = (h1 + h2).view()
    summed = h1.view() + h2.view()

    for field in combined.dtype.names:
        assert summed[field] == approx(combined[field])
    assert summed.sum_of_weights == approx(combined.sum_of_weights)
    assert summed.sum_of_weights_squared == approx(combined.sum_of_weights_squared)
    assert summed.value == approx(combined.value)
    assert summed.variance == approx(combined.variance, nan_ok=True)


def test_mean_view_iadd(mean_pair):
    h1, h2 = mean_pair
    expected = (h1 + h2).view()
    v = h1.view().copy()
    v += h2.view()
    for field in expected.dtype.names:
        assert v[field] == approx(expected[field])


def test_mean_view_zero_count_bins():
    # Both empty -> all-zero result, no NaN and (under filterwarnings=error) no
    # divide/invalid warning.
    empty = bh.Histogram(bh.axis.Integer(0, 3), storage=bh.storage.Mean()).view()
    result = empty + empty
    assert result.count == approx([0, 0, 0])
    assert result.value == approx([0, 0, 0])
    assert result._sum_of_deltas_squared == approx([0, 0, 0])
    assert not np.any(np.isnan(result.value))

    # Empty in one operand only -> result equals the other operand bin-for-bin.
    h = bh.Histogram(bh.axis.Integer(0, 3), storage=bh.storage.Mean())
    h.fill([0, 0, 2], sample=[3.0, 5.0, 7.0])
    left = empty + h.view()
    right = h.view() + empty
    for field in h.view().dtype.names:
        assert left[field] == approx(h.view()[field])
        assert right[field] == approx(h.view()[field])


def test_mean_view_add_accumulator(mean_pair):
    h1, _ = mean_pair
    v = h1.view()
    acc = bh.accumulators.Mean(3, 5.0, 2.0)  # count, value, variance
    result = v + acc

    # Equivalent to merging every bin with a view holding that accumulator.
    other = bh.Histogram(bh.axis.Integer(0, 4), storage=bh.storage.Mean())
    other[...] = [[3.0, 5.0, 2.0]] * 4  # count, value, variance (matches acc)
    expected = v + other.view()
    for field in v.dtype.names:
        assert result[field] == approx(expected[field])


def test_weighted_mean_view_add_accumulator(weighted_mean_pair):
    h1, _ = weighted_mean_pair
    v = h1.view()
    acc = bh.accumulators.WeightedMean(2.0, 1.5, 5.0, 2.0)
    result = v + acc

    # Equivalent to merging every bin with a view holding that accumulator.
    other = bh.Histogram(bh.axis.Integer(0, 4), storage=bh.storage.WeightedMean())
    # sum_of_weights, sum_of_weights_squared, value, variance (matches acc)
    other[...] = [[2.0, 1.5, 5.0, 2.0]] * 4
    expected = v + other.view()
    for field in v.dtype.names:
        assert result[field] == approx(expected[field])


def test_weighted_mean_view_combine_empty_bin():
    # A genuinely empty bin (sum_of_weights == 0) must not introduce NaN.
    h = bh.Histogram(bh.axis.Integer(0, 3), storage=bh.storage.WeightedMean())
    h.fill([0, 2], sample=[3.0, 7.0], weight=[1.0, 1.0])  # bin 1 empty
    empty = bh.Histogram(
        bh.axis.Integer(0, 3), storage=bh.storage.WeightedMean()
    ).view()
    result = h.view() + empty
    for field in h.view().dtype.names:
        assert result[field] == approx(h.view()[field])
    assert not np.any(np.isnan(result.value))


def test_mean_view_scalar_scale():
    h = bh.Histogram(bh.axis.Integer(0, 4), storage=bh.storage.Mean())
    h.fill([0, 0, 1], sample=[2.0, 4.0, 6.0])
    v = h.view()

    for scaled in (v * 3, 3 * v):
        assert scaled.count == approx(v.count)  # weights unchanged
        assert scaled.value == approx(v.value * 3)
        assert scaled._sum_of_deltas_squared == approx(v._sum_of_deltas_squared * 9)

    divided = v / 2
    assert divided.count == approx(v.count)
    assert divided.value == approx(v.value / 2)
    assert divided._sum_of_deltas_squared == approx(v._sum_of_deltas_squared / 4)

    v2 = v.copy()
    v2 *= 3
    assert v2.value == approx(v.value * 3)
    v2 = v.copy()
    v2 /= 2
    assert v2.value == approx(v.value / 2)


def test_weighted_mean_view_scalar_scale(weighted_mean_pair):
    h1, _ = weighted_mean_pair
    v = h1.view()
    scaled = v * 2
    assert scaled.sum_of_weights == approx(v.sum_of_weights)
    assert scaled.sum_of_weights_squared == approx(v.sum_of_weights_squared)
    assert scaled.value == approx(v.value * 2)
    assert scaled._sum_of_weighted_deltas_squared == approx(
        v._sum_of_weighted_deltas_squared * 4
    )


def test_mean_view_reduce_sum():
    h = bh.Histogram(bh.axis.Integer(0, 4), storage=bh.storage.Mean())
    h.fill([0, 0, 1, 3], sample=[1.0, 2.0, 3.0, 9.0])
    v = h.view()

    reduced = v.sum()
    assert isinstance(reduced, bh.accumulators.Mean)

    projected = h.project().view()
    assert reduced.count == approx(projected.count)
    assert reduced.value == approx(projected.value)
    assert reduced._sum_of_deltas_squared == approx(projected._sum_of_deltas_squared)


def test_weighted_mean_view_reduce_sum(weighted_mean_pair):
    h1, _ = weighted_mean_pair
    reduced = h1.view().sum()
    assert isinstance(reduced, bh.accumulators.WeightedMean)

    projected = h1.project().view()
    assert reduced.sum_of_weights == approx(projected.sum_of_weights)
    assert reduced.value == approx(projected.value)
    assert reduced._sum_of_weighted_deltas_squared == approx(
        projected._sum_of_weighted_deltas_squared
    )


def test_mean_view_reduce_keepdims():
    h = bh.Histogram(
        bh.axis.Integer(0, 3), bh.axis.Integer(0, 2), storage=bh.storage.Mean()
    )
    h.fill([0, 0, 1, 2], [0, 1, 1, 0], sample=[1.0, 2.0, 3.0, 4.0])
    v = h.view()

    plain = np.add.reduce(v, axis=0)
    kept = np.add.reduce(v, axis=0, keepdims=True)
    assert plain.shape == (2,)
    assert kept.shape == (1, 2)
    for field in v.dtype.names:
        assert kept[field][0] == approx(plain[field])


# Issue #1143 (B2): mean reductions used to silently ignore reduction
# arguments (out, where, initial, dtype) they cannot honor.
def test_mean_view_reduce_rejects_unsupported_kwargs(mean_pair):
    h1, _ = mean_pair
    v = h1.view()

    with pytest.raises(TypeError, match="axis and keepdims"):
        np.sum(v, dtype=np.float32)
    with pytest.raises(TypeError, match="axis and keepdims"):
        np.sum(v, initial=2.0)
    with pytest.raises(TypeError, match="axis and keepdims"):
        np.sum(v, where=np.zeros(v.shape, dtype=bool))
    with pytest.raises(TypeError, match="out="):
        np.sum(v, out=np.empty(()))


def test_mean_view_reduce_out_matching_dtype(mean_pair):
    h1, _ = mean_pair
    v = h1.view()

    out = np.zeros((), dtype=v.dtype)
    result = np.sum(v, out=out)
    assert np.shares_memory(result, out)

    expected = np.sum(v)
    assert out["count"] == approx(expected.count)
    assert out["value"] == approx(expected.value)
    assert out["_sum_of_deltas_squared"] == approx(expected._sum_of_deltas_squared)


def test_mean_view_rejects_floor_division(mean_pair):
    h1, _ = mean_pair
    v = h1.view()
    with pytest.raises(TypeError):
        v // 2
    with pytest.raises(TypeError):
        2 // v


def test_mean_view_rejects_subtraction(mean_pair):
    h1, h2 = mean_pair
    v1, v2 = h1.view(), h2.view()
    with pytest.raises(TypeError):
        v1 - v2
    with pytest.raises(TypeError):
        v1 - 5


def test_mean_view_rejects_scalar_add(mean_pair):
    h1, _ = mean_pair
    with pytest.raises(TypeError):
        h1.view() + 5


def test_mean_view_rejects_view_product(mean_pair):
    h1, h2 = mean_pair
    v1, v2 = h1.view(), h2.view()
    with pytest.raises(TypeError):
        v1 * v2
    with pytest.raises(TypeError):
        v1 / v2


# Issue #696
def test_view_cumsum():
    h = bh.Histogram(
        bh.axis.Integer(1, 10, underflow=True, overflow=False),
        storage=bh.storage.Weight(),
    )
    h.fill([2, 3], weight=[1.5, 2.5])

    view = h.view()
    c_view = np.cumsum(view)
    assert c_view.value == approx(np.cumsum(view.value))
    assert c_view.variance == approx(np.cumsum(view.variance))
