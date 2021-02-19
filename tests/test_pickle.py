# -*- coding: utf-8 -*-

import copy
import ctypes
import math

import env
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

import boost_histogram as bh

try:
    # Python 2
    from cPickle import dumps, loads
except ImportError:
    from pickle import dumps, loads


ftype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)


def python_convert(x):
    return ftype(x)


def pickle_roundtrip_protocol_2(x):
    return loads(dumps(x, 2))


def pickle_roundtrip_protocol_highest(x):
    return loads(dumps(x, -1))


@pytest.fixture(
    params=(
        pickle_roundtrip_protocol_2,
        pickle_roundtrip_protocol_highest,
        copy.copy,
        copy.deepcopy,
    )
)
def copy_fn(request):
    return request.param


@pytest.mark.parametrize(
    "opts", ({}, {"growth": True}, {"underflow": True, "overflow": True})
)
def test_options(copy_fn, opts):
    orig = bh.axis.Traits(**opts)
    new = copy_fn(orig)
    assert new == orig


@pytest.mark.parametrize(
    "accum,args",
    (
        (bh.accumulators.Sum, (12,)),
        (bh.accumulators.WeightedSum, (1.5, 2.5)),
        (bh.accumulators.Mean, (5, 1.5, 2.5)),
        (bh.accumulators.WeightedMean, (1.5, 2.5, 3.5, 4.5)),
    ),
)
def test_accumulators(accum, args, copy_fn):
    orig = accum(*args)
    new = copy_fn(orig)
    assert new == orig


axes_creations = (
    (bh.axis.Regular, (4, 2, 4), {"underflow": False, "overflow": False}),
    (bh.axis.Regular, (4, 2, 4), {}),
    (bh.axis.Regular, (4, 2, 4), {"growth": True}),
    (bh.axis.Regular, (4, 2, 4), {"transform": bh.axis.transform.log}),
    (bh.axis.Regular, (4, 2, 4), {"transform": bh.axis.transform.sqrt}),
    (bh.axis.Regular, (4, 2, 4), {"transform": bh.axis.transform.Pow(0.5)}),
    (bh.axis.Regular, (4, 2, 4), {"circular": True}),
    (bh.axis.Variable, ([1, 2, 3, 4],), {}),
    (bh.axis.Variable, ([1, 2, 3, 4],), {"circular": True}),
    (bh.axis.Integer, (1, 4), {}),
    (bh.axis.Integer, (1, 4), {"circular": True}),
    (bh.axis.IntCategory, ([1, 2, 3],), {}),
    (bh.axis.IntCategory, ([1, 2, 3],), {"growth": True}),
    (bh.axis.StrCategory, (["1", "2", "3"],), {}),
    (bh.axis.StrCategory, (["1", "2", "3"],), {"growth": True}),
)
raw_axes_creations = ((bh._core.axis.regular_numpy, (4, 2, 4), {}),)


@pytest.mark.parametrize("axis,args,opts", axes_creations + raw_axes_creations)
def test_axes(axis, args, opts, copy_fn):
    orig = axis(*args, **opts)
    new = copy_fn(orig)
    assert new == orig
    np.testing.assert_array_equal(new.centers, orig.centers)


@pytest.mark.parametrize("axis,args,opts", axes_creations)
def test_metadata_str(axis, args, opts, copy_fn):
    orig = axis(*args, **opts)
    orig.metadata = "foo"
    new = copy_fn(orig)
    assert new.metadata == orig.metadata
    new.metadata = orig.metadata
    assert new == orig
    np.testing.assert_array_equal(new.centers, orig.centers)


# Special test: Deepcopy should change metadata id, copy should not
def test_compare_copy_axis(metadata):
    orig = bh.axis.Regular(4, 0, 2, metadata=metadata)
    new = copy.copy(orig)
    dnew = copy.deepcopy(orig)

    assert orig.metadata is new.metadata
    assert orig.metadata == dnew.metadata
    if metadata is not copy.copy(metadata):
        assert orig.metadata is not dnew.metadata


# Special test: Deepcopy should change metadata id, copy should not
def test_compare_copy_hist(metadata):
    orig = bh.Histogram(bh.axis.Regular(4, 0, 2, metadata=metadata))
    new = copy.copy(orig)
    dnew = copy.deepcopy(orig)

    assert orig.axes[0].metadata is new.axes[0].metadata
    assert orig.axes[0].metadata == dnew.axes[0].metadata
    if metadata is not copy.copy(metadata):
        assert orig.axes[0].metadata is not dnew.axes[0].metadata


@pytest.mark.parametrize("axis,args,opts", axes_creations)
def test_metadata_any(axis, args, opts, copy_fn):
    orig = axis(*args, **opts)
    orig.metadata = (1, 2, 3)
    new = copy_fn(orig)
    assert new.metadata == orig.metadata
    new.metadata = orig.metadata
    assert new == orig


@pytest.mark.benchmark(group="histogram-pickling")
@pytest.mark.parametrize(
    "storage, extra",
    (
        (bh.storage.AtomicInt64, {}),
        (bh.storage.Int64, {}),
        (bh.storage.Unlimited, {}),
        (bh.storage.Unlimited, {"weight"}),
        (bh.storage.Double, {"weight"}),
        (bh.storage.Weight, {"weight"}),
        (bh.storage.Mean, {"sample"}),
        (bh.storage.WeightedMean, {"weight", "sample"}),
    ),
)
def test_storage(benchmark, copy_fn, storage, extra):
    n = 1000
    hist = bh.Histogram(bh.axis.Integer(0, n), storage=storage())
    x = np.arange(2 * (n + 2)) % (n + 2) - 1
    if extra == {}:
        hist.fill(x)
    elif extra == {"weight"}:
        hist.fill(x, weight=np.arange(2 * n + 4) + 1)
    elif extra == {"sample"}:
        hist.fill(x, sample=np.arange(2 * n + 4) + 1)
    else:
        hist.fill(x, weight=np.arange(2 * n + 4) + 1, sample=np.arange(2 * n + 4) + 1)

    new = benchmark(copy_fn, hist)
    assert_array_equal(hist.view(True), new.view(True))
    assert new == hist


def test_histogram_regular(copy_fn):
    hist = bh.Histogram(bh.axis.Regular(4, 1, 2), bh.axis.Regular(8, 3, 6))

    new = copy_fn(hist)
    assert hist == new


def test_histogram_fancy(copy_fn):
    hist = bh.Histogram(bh.axis.Regular(4, 1, 2), bh.axis.Integer(0, 6))

    new = copy_fn(hist)
    assert hist == new


def test_histogram_metadata(copy_fn, metadata):

    hist = bh.Histogram(bh.axis.Regular(4, 1, 2, metadata=metadata))
    new = copy_fn(hist)
    assert hist == new


def test_numpy_edge(copy_fn):
    ax1 = bh._core.axis.regular_numpy(10, 0, 1)
    ax2 = copy_fn(ax1)

    # stop defaults to 0, so this fails if the copy fails
    assert ax1 == ax2
    assert ax1.index(1) == ax2.index(1)
    assert ax2.index(1) == 9


@pytest.mark.skipif(env.PYPY, reason="Not remotely supported on PyPY, hangs forever")
@pytest.mark.parametrize("mod", (np, math))
def test_pickle_transforms(mod, copy_fn):
    ax1 = bh.axis.Regular(
        100,
        1,
        100,
        transform=bh.axis.transform.Function(mod.log, mod.exp, convert=python_convert),
    )
    ax2 = copy_fn(ax1)
    ax3 = bh.axis.Regular(100, 1, 100, transform=bh.axis.transform.log)

    assert ax1 == ax2
    assert_array_equal(ax1.centers, ax2.centers)
    assert_almost_equal(ax2.centers, ax3.centers, decimal=10)


def test_hist_axes_reference(copy_fn):
    h = bh.Histogram(bh.axis.Regular(10, 0, 1, metadata=1))
    h.axes[0].metadata = 2

    h2 = copy_fn(h)

    assert h2._hist is not h._hist
    assert h2.axes[0] is not h.axes[0]

    assert h2.axes[0].metadata == 2

    h.axes[0].metadata = 3
    assert h2._axis(0).metadata == 2
    assert h2.axes[0].metadata == 2


def test_hist_axes_reference_arbitrary(copy_fn):
    h = bh.Histogram(bh.axis.Regular(10, 0, 1))
    h.other = 2

    h2 = copy_fn(h)

    assert h2._hist is not h._hist

    assert h2.other == 2

    h.other = 3
    assert h2.other == 2


def test_hist_reference_arbitrary(copy_fn):
    h = bh.Histogram(bh.axis.Regular(10, 0, 1))
    h.axes[0].other = 2

    h2 = copy_fn(h)

    assert h2._hist is not h._hist
    assert h2.axes[0] is not h.axes[0]

    assert h2.axes[0].other == 2

    h.axes[0].other = 3
    assert h2._axis(0).other == 2
    assert h2.axes[0].other == 2


def test_axis_wrapped(copy_fn):
    ax = bh.axis.Regular(10, 0, 2)
    ax2 = copy_fn(ax)

    assert ax._ax is not ax2._ax


def test_trans_wrapped(copy_fn):
    tr = bh.axis.transform.Pow(2)
    tr2 = copy_fn(tr)

    assert tr._this is not tr2._this


# Testing #342
def test_cloudpickle():
    cloudpickle = pytest.importorskip("cloudpickle")
    h = bh.Histogram(bh.axis.Regular(50, 0, 20))
    h.fill([1, 2, 3, 4, 5])
    h2 = loads(cloudpickle.dumps(h))

    assert h == h2
    assert h is not h2
