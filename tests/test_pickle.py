import pytest
import numpy as np
from numpy.testing import assert_array_equal

import ctypes

ftype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)


def python_convert(x):
    return ftype(x)


try:
    # Python 2
    from cPickle import loads, dumps
except ImportError:
    from pickle import loads, dumps

import copy
import math

import boost_histogram as bh

copy_fns = (
    lambda x: loads(dumps(x, 2)),
    lambda x: loads(dumps(x, -1)),
    copy.copy,
    copy.deepcopy,
)

accumulators = (
    (bh.accumulators.Sum, (12,)),
    (bh.accumulators.WeightedSum, (1.5, 2.5)),
    (bh.accumulators.Mean, (5, 1.5, 2.5)),
    (bh.accumulators.WeightedMean, (1.5, 2.5, 3.5, 4.5)),
)

storages = (
    bh.storage.AtomicInt,
    bh.storage.Double,
    bh.storage.Int,
    bh.storage.Mean,
    bh.storage.Unlimited,
    bh.storage.Weight,
    bh.storage.WeightedMean,
)


@pytest.mark.parametrize("copy_fn", copy_fns)
@pytest.mark.parametrize(
    "opts", ({}, {"growth": True}, {"underflow": True, "overflow": True})
)
def test_options(copy_fn, opts):
    orig = bh.axis.options(**opts)
    new = copy_fn(orig)
    assert new == orig


@pytest.mark.parametrize("accum,args", accumulators)
@pytest.mark.parametrize("copy_fn", copy_fns)
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
    (bh._core.axis.regular_numpy, (4, 2, 4), {}),
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


@pytest.mark.parametrize("axis,args,opts", axes_creations)
@pytest.mark.parametrize("copy_fn", copy_fns)
def test_axes(axis, args, opts, copy_fn):
    orig = axis(*args, metadata=None, **opts)
    new = copy_fn(orig)
    assert new == orig
    np.testing.assert_array_equal(new.centers, orig.centers)


@pytest.mark.parametrize("axis,args,opts", axes_creations)
@pytest.mark.parametrize("copy_fn", copy_fns)
def test_metadata_str(axis, args, opts, copy_fn):
    orig = axis(*args, metadata="foo", **opts)
    new = copy_fn(orig)
    assert new.metadata == orig.metadata
    new.metadata = orig.metadata
    assert new == orig
    np.testing.assert_array_equal(new.centers, orig.centers)


# Special test: Deepcopy should change metadata id, copy should not
@pytest.mark.parametrize("metadata", ({1: 2}, [1, 2, 3]))
def test_compare_copy_axis(metadata):
    orig = bh.axis.Regular(4, 0, 2, metadata=metadata)
    new = copy.copy(orig)
    dnew = copy.deepcopy(orig)

    assert orig.metadata is new.metadata
    assert orig.metadata == dnew.metadata
    assert orig.metadata is not dnew.metadata


# Special test: Deepcopy should change metadata id, copy should not
@pytest.mark.parametrize("metadata", ({1: 2}, [1, 2, 3]))
def test_compare_copy_hist(metadata):
    orig = bh.Histogram(bh.axis.Regular(4, 0, 2, metadata=metadata))
    new = copy.copy(orig)
    dnew = copy.deepcopy(orig)

    assert orig.axes[0].metadata is new.axes[0].metadata
    assert orig.axes[0].metadata == dnew.axes[0].metadata
    assert orig.axes[0].metadata is not dnew.axes[0].metadata


@pytest.mark.parametrize("axis,args,opts", axes_creations)
@pytest.mark.parametrize("copy_fn", copy_fns)
def test_metadata_any(axis, args, opts, copy_fn):
    orig = axis(*args, metadata=(1, 2, 3), **opts)
    new = copy_fn(orig)
    assert new.metadata == orig.metadata
    new.metadata = orig.metadata
    assert new == orig


@pytest.mark.parametrize("copy_fn", copy_fns)
@pytest.mark.parametrize("storage", storages)
def test_storage_int(copy_fn, storage):
    storage = storage()

    new = copy_fn(storage)
    assert storage == new


@pytest.mark.parametrize("copy_fn", copy_fns)
def test_histogram_regular(copy_fn):
    hist = bh.Histogram(bh.axis.Regular(4, 1, 2), bh.axis.Regular(8, 3, 6))

    new = copy_fn(hist)
    assert hist == new


@pytest.mark.parametrize("copy_fn", copy_fns)
def test_histogram_fancy(copy_fn):
    hist = bh.Histogram(bh.axis.Regular(4, 1, 2), bh.axis.Integer(0, 6))

    new = copy_fn(hist)
    assert hist == new


@pytest.mark.parametrize("copy_fn", copy_fns)
@pytest.mark.parametrize("metadata", ("This", (1, 2, 3)))
def test_histogram_metadata(copy_fn, metadata):

    hist = bh.Histogram(bh.axis.Regular(4, 1, 2, metadata=metadata))
    new = copy_fn(hist)
    assert hist == new


@pytest.mark.parametrize("copy_fn", copy_fns)
def test_numpy_edge(copy_fn):
    ax1 = bh._core.axis.regular_numpy(10, 0, 1, None)
    ax2 = copy_fn(ax1)

    # stop defaults to 0, so this fails if the copy fails
    assert ax1 == ax2
    assert ax1.index(1) == ax2.index(1)
    assert ax2.index(1) == 9


@pytest.mark.parametrize("mod", (np, math))
@pytest.mark.parametrize("copy_fn", copy_fns)
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
    assert_array_equal(ax2.centers, ax3.centers)
