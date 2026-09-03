from __future__ import annotations

import math
import operator

import numpy as np
import pytest
from pytest import approx

import boost_histogram as bh


@pytest.mark.parametrize(
    "storage",
    [bh.storage.Int64, bh.storage.Double, bh.storage.AtomicInt64, bh.storage.Unlimited],
)
def test_setting(storage):
    h = bh.Histogram(bh.axis.Regular(10, 0, 1), storage=storage())

    h[0] = 2
    h[1] = 3
    h[-1] = 5

    assert h[0] == 2
    assert h[1] == 3
    assert h[9] == 5

    assert h.view() == approx([2, 3, 0, 0, 0, 0, 0, 0, 0, 5])


def test_setting_weight():
    h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.Weight())

    h.fill([0.3, 0.3, 0.4, 1.2])

    assert h[0] == bh.accumulators.WeightedSum(3, 3)
    assert h[1] == bh.accumulators.WeightedSum(1, 1)

    h[0] = bh.accumulators.WeightedSum(value=2, variance=2)
    assert h[0] == bh.accumulators.WeightedSum(2, 2)

    a = h.view()

    assert a[0] == h[0]

    b = np.asarray(h)

    assert b["value"][0] == h[0].value
    assert b["variance"][0] == h[0].variance

    h[0] = bh.accumulators.WeightedSum(value=3, variance=1)

    assert h[0].value == 3
    assert h[0].variance == 1

    assert a[0] == h[0]

    assert b["value"][0] == h[0].value
    assert b["variance"][0] == h[0].variance

    assert b[0]["value"] == a[0]["value"]
    assert b[0]["variance"] == a[0]["variance"]

    assert b["value"][0] == a["value"][0]
    assert b["variance"][0] == a["variance"][0]

    assert a.view().value == approx(b.view()["value"])
    assert a.view().variance == approx(b.view()["variance"])


def test_sum_weight():
    h = bh.Histogram(bh.axis.Integer(0, 10), storage=bh.storage.Weight())
    h.fill([1, 2, 3, 3, 3, 4, 5])
    v = h.view().copy()
    res = np.sum(v)
    hres = h.sum()
    assert res.value == hres.value == 7
    assert res.variance == hres.variance == 7

    v2 = v + v
    h2 = h + h

    assert h2.view().value == approx(v2.value)
    assert h2.view().variance == approx(v2.variance)


def test_setting_profile():
    h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.Mean())

    h.fill([0.3, 0.3, 0.4, 1.2, 1.6], sample=[1, 2, 3, 4, 4])

    assert h[0] == bh.accumulators.Mean(count=3, value=2, variance=1)
    assert h[1] == bh.accumulators.Mean(count=2, value=4, variance=0)

    h[0] = bh.accumulators.Mean(count=12, value=11, variance=10)
    assert h[0] == bh.accumulators.Mean(count=12, value=11, variance=10)

    a = h.view()

    assert a[0] == h[0]

    b = np.asarray(h)

    assert b["value"][0] == h[0].value
    assert b["count"][0] == h[0].count
    assert b["_sum_of_deltas_squared"][0] == h[0]._sum_of_deltas_squared

    h[0] = bh.accumulators.Mean(count=6, value=3, variance=2)
    assert h[0].count == 6
    assert h[0].value == 3
    assert h[0].variance == 2

    assert a[0] == h[0]

    assert b["value"][0] == h[0].value
    assert b["count"][0] == h[0].count
    assert b["_sum_of_deltas_squared"][0] == h[0]._sum_of_deltas_squared

    assert b[0]["value"] == a[0]["value"]
    assert b[0]["count"] == a[0]["count"]
    assert b[0]["_sum_of_deltas_squared"] == a[0]["_sum_of_deltas_squared"]

    assert b[0]["value"] == a["value"][0]
    assert b[0]["count"] == a["count"][0]
    assert b[0]["_sum_of_deltas_squared"] == a["_sum_of_deltas_squared"][0]

    assert a.view().value == approx(b.view()["value"])
    assert a.view().count == approx(b.view()["count"])
    assert a.view()._sum_of_deltas_squared == approx(b.view()["_sum_of_deltas_squared"])


def test_mean_storage_rejects_string_sample():
    h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.Mean())

    with pytest.raises(
        TypeError,
        match=r"Sample key-argument needs to be a number or a sequence, str given\.",
    ):
        h.fill([0.3], sample="abc")


def test_setting_weighted_profile():
    h = bh.Histogram(bh.axis.Regular(10, 0, 10), storage=bh.storage.WeightedMean())

    h.fill([0.3, 0.3, 0.4, 1.2, 1.6], sample=[1, 2, 3, 4, 4], weight=[1, 1, 1, 1, 2])

    assert h[0] == bh.accumulators.WeightedMean(
        sum_of_weights=3, sum_of_weights_squared=3, value=2, variance=1
    )
    assert h[1] == bh.accumulators.WeightedMean(
        sum_of_weights=3, sum_of_weights_squared=5, value=4, variance=0
    )

    h[0] = bh.accumulators.WeightedMean(
        sum_of_weights=12, sum_of_weights_squared=15, value=11, variance=10
    )
    assert h[0] == bh.accumulators.WeightedMean(
        sum_of_weights=12, sum_of_weights_squared=15, value=11, variance=10
    )

    a = h.view()

    assert a[0] == h[0]

    b = np.asarray(h)

    assert b["value"][0] == h[0].value
    assert b["sum_of_weights"][0] == h[0].sum_of_weights
    assert b["sum_of_weights_squared"][0] == h[0].sum_of_weights_squared
    assert (
        b["_sum_of_weighted_deltas_squared"][0] == h[0]._sum_of_weighted_deltas_squared
    )

    h[0] = bh.accumulators.WeightedMean(
        sum_of_weights=6, sum_of_weights_squared=12, value=3, variance=2
    )

    assert a[0] == h[0]

    assert h[0].value == 3
    assert h[0].variance == 2
    assert h[0].sum_of_weights == 6
    assert h[0].sum_of_weights_squared == 12
    assert h[0]._sum_of_weighted_deltas_squared == 8

    assert b["value"][0] == h[0].value
    assert b["sum_of_weights"][0] == h[0].sum_of_weights
    assert b["sum_of_weights_squared"][0] == h[0].sum_of_weights_squared
    assert (
        b["_sum_of_weighted_deltas_squared"][0] == h[0]._sum_of_weighted_deltas_squared
    )

    assert b[0]["value"] == a[0]["value"]
    assert b[0]["sum_of_weights"] == a[0]["sum_of_weights"]
    assert b[0]["sum_of_weights_squared"] == a[0]["sum_of_weights_squared"]
    assert (
        b[0]["_sum_of_weighted_deltas_squared"]
        == a[0]["_sum_of_weighted_deltas_squared"]
    )

    assert b[0]["value"] == a["value"][0]
    assert b[0]["sum_of_weights"] == a["sum_of_weights"][0]
    assert b[0]["sum_of_weights_squared"] == a["sum_of_weights_squared"][0]
    assert (
        b[0]["_sum_of_weighted_deltas_squared"]
        == a["_sum_of_weighted_deltas_squared"][0]
    )

    assert a.view().value == approx(b.view()["value"])
    assert a.view().sum_of_weights == approx(b.view()["sum_of_weights"])
    assert a.view().sum_of_weights_squared == approx(b.view()["sum_of_weights_squared"])
    assert a.view()._sum_of_weighted_deltas_squared == approx(
        b.view()["_sum_of_weighted_deltas_squared"]
    )


# Issue #486
def test_modify_weights_by_view():
    bins = [0, 1, 2]
    hist = bh.Histogram(bh.axis.Variable(bins), storage=bh.storage.Weight())
    yields = [3, 4]
    var = [0.1, 0.2]
    hist[...] = np.stack([yields, var], axis=-1)

    hist.view().value /= 2

    assert hist.view().value[0] == pytest.approx(1.5)
    assert hist.view().value[1] == pytest.approx(2)


# Issue #531
def test_summing_mean_storage():
    np.random.seed(42)
    values = np.random.normal(loc=1.3, scale=0.1, size=1000)
    samples = np.random.normal(loc=1.3, scale=0.1, size=1000)

    h1 = bh.Histogram(bh.axis.Regular(20, -1, 3), storage=bh.storage.Mean())
    h1.fill(values, sample=samples)

    h2 = bh.Histogram(bh.axis.Regular(1, -1, 3), storage=bh.storage.Mean())
    h2.fill(values, sample=samples)

    s1 = h1.sum()
    s2 = h2.sum()

    assert s1.value == approx(s2.value)
    assert s1.count == approx(s2.count)
    assert s1.variance == approx(s2.variance)


# Issue #531
def test_summing_weighted_mean_storage():
    np.random.seed(42)
    values = np.random.normal(loc=1.3, scale=0.1, size=1000)
    samples = np.random.normal(loc=1.3, scale=0.1, size=1000)
    weights = np.random.uniform(0.1, 5, size=1000)

    h1 = bh.Histogram(bh.axis.Regular(20, -1, 3), storage=bh.storage.WeightedMean())
    h1.fill(values, sample=samples, weight=weights)

    h2 = bh.Histogram(bh.axis.Regular(1, -1, 3), storage=bh.storage.WeightedMean())
    h2.fill(values, sample=samples, weight=weights)

    s1 = h1.sum()
    s2 = h2.sum()

    assert s1.value == approx(s2.value)
    assert s1.sum_of_weights == approx(s2.sum_of_weights)
    assert s1.sum_of_weights_squared == approx(s2.sum_of_weights_squared)
    assert s1.variance == approx(s2.variance)


# Raised on Gitter
def test_UHI_variance_counts():
    h = bh.Histogram(
        bh.axis.Regular(bins=1, start=0, stop=1), storage=bh.storage.WeightedMean()
    )
    h.fill(0.5, sample=[1], weight=[0.5])
    h.fill(0.5, sample=[2], weight=[0.4])
    assert not math.isnan(h.variances()[0])

    h = bh.Histogram(
        bh.axis.Regular(bins=1, start=0, stop=1), storage=bh.storage.WeightedMean()
    )
    h.fill(0.5, sample=[1], weight=[0.5])
    h.fill(0.5, sample=[1], weight=[0.5])
    assert not math.isnan(h.variances()[0])


@pytest.mark.parametrize(
    "storage",
    [
        bh.storage.Int64,
        bh.storage.Double,
        bh.storage.AtomicInt64,
        bh.storage.Unlimited,
        bh.storage.Weight,
        bh.storage.Mean,
        bh.storage.WeightedMean,
    ],
)
def test_empty_axis_histogram(storage):
    h2d = bh.Histogram(
        bh.axis.Regular(10, 0, 10),
        bh.axis.StrCategory([], growth=True),
        storage=storage(),
    )
    assert h2d.sum() == h2d.storage_type.accumulator()
    assert h2d.sum(flow=True) == h2d.storage_type.accumulator()


# Issue #971
def test_non_uniform_rebin_with_weights():
    # 1D
    h = bh.Histogram(bh.axis.Regular(20, 1, 5), storage=bh.storage.Weight())
    h.fill([1.1, 2.2, 3.3, 4.4])

    rslt = np.array(
        [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (0.0, 0.0), (1.0, 1.0)],
        dtype=[("value", "<f8"), ("variance", "<f8")],
    )

    hs = h[{0: slice(None, None, bh.rebin(4))}]
    assert hs.view() == approx(rslt)

    hs = h[{0: bh.rebin(4)}]
    assert hs.view() == approx(rslt)

    hs = h[{0: bh.rebin(groups=[1, 2, 3, 14])}]
    assert hs.view() == approx(
        np.array(
            [(1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (3.0, 3.0)],
            dtype=[("value", "<f8"), ("variance", "<f8")],
        )
    )
    assert hs.axes.edges[0] == approx([1.0, 1.2, 1.6, 2.2, 5.0])

    # nD
    h = bh.Histogram(
        bh.axis.Regular(20, 1, 3),
        bh.axis.Regular(30, 1, 3),
        bh.axis.Regular(40, 1, 3),
        storage=bh.storage.Weight(),
    )

    assert h[{0: np.s_[:: bh.rebin(groups=[1, 2, 17])]}].axes.size == (3, 30, 40)
    assert h[{1: np.s_[:: bh.rebin(groups=[1, 2, 27])]}].axes.size == (20, 3, 40)
    assert h[{2: np.s_[:: bh.rebin(groups=[1, 2, 37])]}].axes.size == (20, 30, 3)
    assert np.all(
        np.isclose(
            h[{0: np.s_[:: bh.rebin(groups=[1, 2, 17])]}].axes[0].edges,
            [1.0, 1.1, 1.3, 3.0],
        )
    )
    assert np.all(
        np.isclose(
            h[{1: np.s_[:: bh.rebin(groups=[1, 2, 27])]}].axes[1].edges,
            [1.0, 1.06666667, 1.2, 3.0],
        )
    )
    assert np.all(
        np.isclose(
            h[{2: np.s_[:: bh.rebin(groups=[1, 2, 37])]}].axes[2].edges,
            [1.0, 1.05, 1.15, 3.0],
        )
    )

    assert h[
        {
            0: np.s_[:: bh.rebin(groups=[1, 2, 17])],
            2: np.s_[:: bh.rebin(groups=[1, 2, 37])],
        }
    ].axes.size == (3, 30, 3)
    assert np.all(
        np.isclose(
            h[
                {
                    0: np.s_[:: bh.rebin(groups=[1, 2, 17])],
                    2: np.s_[:: bh.rebin(groups=[1, 2, 37])],
                }
            ]
            .axes[0]
            .edges,
            [1.0, 1.1, 1.3, 3],
        )
    )
    assert np.all(
        np.isclose(
            h[
                {
                    0: np.s_[:: bh.rebin(groups=[1, 2, 17])],
                    2: np.s_[:: bh.rebin(groups=[1, 2, 37])],
                }
            ]
            .axes[2]
            .edges,
            [1.0, 1.05, 1.15, 3.0],
        )
    )


def _filled_multi_cell(nelem, start=1):
    """A 1D MultiCell histogram filled so that bin i holds an arange-based vector."""
    h = bh.Histogram(bh.axis.Regular(5, 0, 5), storage=bh.storage.MultiCell(nelem))
    x = np.array([1, 2, 3])
    weights = (np.arange(3 * nelem) + start).reshape(3, nelem)
    h.fill(x, weight=weights)
    return h


@pytest.mark.parametrize("nelem", [1, 3])
def test_multi_cell_arithmetic(nelem):
    h = _filled_multi_cell(nelem)
    base = h.view().copy()

    # Histogram + Histogram (and the in-place form)
    assert (h + h).view() == approx(base * 2)

    h_iadd = _filled_multi_cell(nelem)
    h_iadd += h
    assert h_iadd.view() == approx(base * 2)

    # Histogram - Histogram (element-wise on the cells), including the in-place
    # form and a result with negative cells
    assert (h - h).view() == approx(base * 0)
    assert (h + h - h).view() == approx(base)
    assert (h - h - h).view() == approx(-base)

    h_isub = _filled_multi_cell(nelem)
    h_isub -= h
    assert h_isub.view() == approx(base * 0)

    # Scalar multiplication / division (both orders and in-place)
    assert (h * 2).view() == approx(base * 2)
    assert (2 * h).view() == approx(base * 2)
    assert (h / 2).view() == approx(base / 2)

    h_imul = _filled_multi_cell(nelem)
    h_imul *= 2
    assert h_imul.view() == approx(base * 2)

    h_idiv = _filled_multi_cell(nelem)
    h_idiv /= 2
    assert h_idiv.view() == approx(base / 2)


@pytest.mark.parametrize("nelem", [1, 3])
def test_multi_cell_equality(nelem):
    h1 = _filled_multi_cell(nelem)
    h2 = _filled_multi_cell(nelem)
    # Exercise both == and != directly (hence the explicit non-simplified forms)
    assert h1 == h2
    assert not (h1 != h2)  # noqa: SIM202

    h2 += h1
    assert h1 != h2
    assert not (h1 == h2)  # noqa: SIM201


@pytest.mark.parametrize("op", [operator.add, operator.sub], ids=["add", "sub"])
def test_multi_cell_mismatched_nelem(op):
    h3 = _filled_multi_cell(3)
    h2 = _filled_multi_cell(2)
    with pytest.raises(ValueError, match="size does not match"):
        op(h3, h2)


# Issue #1128
@pytest.mark.parametrize(
    ("storage", "fill_kwargs"),
    [
        (bh.storage.Weight(), {}),
        (bh.storage.Mean(), {"sample": [1]}),
        (bh.storage.WeightedMean(), {"sample": [1], "weight": [1]}),
    ],
    ids=["Weight", "Mean", "WeightedMean"],
)
def test_unsupported_histogram_subtraction(storage, fill_kwargs):
    # Storages whose C++ backend has no __isub__ must raise a clear TypeError
    # rather than leaking the dunder name via AttributeError. (MultiCell does
    # support subtraction; see test_multi_cell_arithmetic.)
    def mk():
        h = bh.Histogram(bh.axis.Regular(5, 0, 5), storage=storage)
        h.fill([1], **fill_kwargs)
        return h

    name = type(storage).__name__
    with pytest.raises(
        TypeError,
        match=rf"{name} storage does not support the '-' operation",
    ):
        mk() - mk()


# Issue #1143 (B6a)
def test_multi_cell_rebin_groups():
    h = bh.Histogram(bh.axis.Regular(4, 0, 4), storage=bh.storage.MultiCell(2))
    h.fill([0.5, 1.5, 2.5, 3.5], weight=[[1, 10], [2, 20], [3, 30], [4, 40]])
    h.view(flow=True)[:, 0] = [100, 1000]
    h.view(flow=True)[:, -1] = [200, 2000]

    hs = h[bh.rebin(groups=[2, 2])]
    assert hs.storage.nelem == 2
    assert hs.view() == approx(np.array([[3, 7], [30, 70]]))
    assert hs.view(flow=True) == approx(
        np.array([[100, 3, 7, 200], [1000, 30, 70, 2000]])
    )


def test_multi_cell_rebin_groups_2d():
    # Rebin a non-leading axis: the cell index is the leading view dimension,
    # so the axis offset must be applied when combining groups.
    h = bh.Histogram(
        bh.axis.Regular(2, 0, 2),
        bh.axis.Regular(4, 0, 4),
        storage=bh.storage.MultiCell(2),
    )
    h.fill(
        [0.5, 0.5, 0.5, 0.5, 1.5],
        [0.5, 1.5, 2.5, 3.5, 0.5],
        weight=[[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]],
    )

    hs = h[:, bh.rebin(groups=[2, 2])]
    assert hs.storage.nelem == 2
    assert hs.view() == approx(np.array([[[3, 7], [5, 0]], [[30, 70], [50, 0]]]))
    # The first axis is untouched
    assert hs.view().sum(axis=(1, 2)) == approx(h.view().sum(axis=(1, 2)))


@pytest.mark.parametrize("nelem", [1, 3])
def test_multi_cell_reset(nelem):
    h = _filled_multi_cell(nelem)
    assert h.view().sum() != 0
    h.reset()
    assert h.view() == approx(np.zeros_like(h.view()))
    assert h.view(flow=True) == approx(np.zeros_like(h.view(flow=True)))


# Issue #1129
def test_multi_cell_empty_axis():
    # An empty categorical axis means zero bins, but the sum is still reported
    # as a zero-filled vector of length nelem for consistency with non-empty
    # histograms (rather than an empty list).
    h = bh.Histogram(
        bh.axis.Regular(10, 0, 10),
        bh.axis.StrCategory([], growth=True),
        storage=bh.storage.MultiCell(3),
    )
    assert h.sum() == [0.0, 0.0, 0.0]
    assert h.sum(flow=True) == [0.0, 0.0, 0.0]


def test_multi_cell():
    x = np.array([1, 2])
    y = np.array([0, 1])
    weights = np.array([[1, 2, 3], [4, 5, 6]])
    h = bh.Histogram(bh.axis.Regular(5, 0, 5), storage=bh.storage.MultiCell(3))

    # Repr must contain MultiCell(3)
    repr_h = repr(h)
    assert "MultiCell(3)" in repr_h

    # Can access number of elements
    assert h.storage.nelem == 3

    # Can use pattern matching
    match h:
        case bh.Histogram(storage=bh.storage.MultiCell(nelem)):
            assert nelem == 3
        case _:
            raise AssertionError("Can't pattern match")

    # Filling 1-Dim
    h.fill(x, weight=weights)
    assert h[1] == approx([1, 2, 3])
    assert h[2] == approx([4, 5, 6])

    h = bh.Histogram(
        bh.axis.Regular(5, 0, 5),
        bh.axis.Regular(3, 0, 3),
        storage=bh.storage.MultiCell(3),
    )

    # Filling 2-Dim
    h.fill(x, y, weight=weights)
    assert h[1, 0] == approx([1, 2, 3])
    assert h[2, 1] == approx([4, 5, 6])

    # View and values

    x = np.array([1, 2, 3, 4])
    y = np.array([2, 2, 0, 1])
    weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    h.fill(x, y, weight=weights)

    expected_view_with_flow = np.array(
        [
            # weight index 0
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 4.0, 4.0, 0.0],
                [0.0, 7.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 10.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            # weight index 1
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 5.0, 5.0, 0.0],
                [0.0, 8.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 11.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            # weight index 2
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 6.0, 6.0, 0.0],
                [0.0, 9.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 12.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ]
    )

    expected_view = expected_view_with_flow[:, 1:-1, 1:-1]

    assert h.view() == approx(expected_view)
    assert h.view(flow=True) == approx(expected_view_with_flow)

    assert h.values() == approx(expected_view)
    assert h.values(flow=True) == approx(expected_view_with_flow)

    # Modify view
    expected_view[0, 1, 0] = 10
    expected_view_with_flow[0, 2, 1] = 10
    h.view()[0, 1, 0] = 10

    assert h.view() == approx(expected_view)
    assert h.view(flow=True) == approx(expected_view_with_flow)

    assert h.values() == approx(expected_view)
    assert h.values(flow=True) == approx(expected_view_with_flow)

    # Slice histogram
    ## via reduce() (only use real slices)
    assert h[1:3, 1:3].view() == approx(expected_view[:, 1:3, 1:3])

    ## Without reduce() (only slices and single elements)
    assert h[2, 1:3].view() == approx(expected_view[:, 2, 1:3])

    # Project histogram
    assert h.project(1).view() == approx(
        np.sum(expected_view_with_flow, axis=1)[:, 1:-1]
    )
    assert h.project(0).view() == approx(
        np.sum(expected_view_with_flow, axis=2)[:, 1:-1]
    )

    # Sum histogram
    assert h.sum() == approx(np.sum(expected_view, axis=(1, 2)))
    assert h.sum(flow=True) == approx(np.sum(expected_view_with_flow, axis=(1, 2)))

    # __setitem__ for histogram
    sub_array_to_set = np.array([[20, 30], [21, 31], [22, 32]])
    expected_view[:, 2:4, 1] = sub_array_to_set
    h[2:4, 1] = sub_array_to_set
    assert h.view() == approx(expected_view)

    sub_array_to_set = np.array(
        [
            [[40, 41, 42], [43, 44, 45]],
            [[50, 51, 52], [53, 54, 55]],
            [[60, 61, 62], [63, 64, 65]],
        ]
    )
    expected_view[:, 2:4, 0:3] = sub_array_to_set
    h[2:4, 0:3] = sub_array_to_set
    assert h.view() == approx(expected_view)


def test_multi_cell_bad_nelem():
    with pytest.raises(ValueError, match="nelem"):
        bh.storage.MultiCell(-1)
    with pytest.raises(ValueError, match="nelem"):
        bh.storage.MultiCell(0)


def test_multi_cell_fill_wrong_width():
    h = bh.Histogram(bh.axis.Regular(3, 0, 1), storage=bh.storage.MultiCell(2))
    with pytest.raises(ValueError, match="2"):
        h.fill([0.1, 0.5], weight=[[1, 2, 3], [4, 5, 6]])


def test_multi_cell_empty():
    h = bh.Histogram(bh.axis.Regular(3, 0, 1), storage=bh.storage.MultiCell(2))
    assert h.empty()
    assert h.empty(flow=True)

    h.fill([0.1], weight=[[1, 2]])
    assert not h.empty()
    assert not h.empty(flow=True)

    h2 = bh.Histogram(bh.axis.Regular(3, 0, 1), storage=bh.storage.MultiCell(2))
    h2.fill([-1], weight=[[1, 2]])
    assert h2.empty()
    assert not h2.empty(flow=True)
