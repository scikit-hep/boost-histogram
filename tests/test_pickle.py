import pytest

try:
    # Python 2
    from cPickle import loads, dumps
except ImportError:
    from pickle import loads, dumps

import copy

import boost_histogram as bh

copy_fns = (
    lambda x: loads(dumps(x, 2)),
    lambda x: loads(dumps(x, -1)),
    copy.copy,
    copy.deepcopy,
)

accumulators = (
    (bh.accumulators.sum, (12,)),
    (bh.accumulators.weighted_sum, (1.5, 2.5)),
    (bh.accumulators.mean, (5, 1.5, 2.5)),
    (bh.accumulators.weighted_mean, (1.5, 2.5, 3.5, 4.5)),
)

storages = (
    bh.storage.atomic_int,
    bh.storage.double,
    bh.storage.int,
    bh.storage.mean,
    bh.storage.unlimited,
    bh.storage.weight,
    bh.storage.weighted_mean,
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
    (bh.axis.regular, (4, 2, 4), {"underflow": False, "overflow": False}),
    (bh.axis.regular, (4, 2, 4), {}),
    (bh.axis.regular, (4, 2, 4), {"growth": True}),
    (bh.axis.regular_log, (4, 2, 4), {}),
    (bh.axis.regular_sqrt, (4, 2, 4), {}),
    (bh.axis.regular_pow, (4, 2, 4, 0.5), {}),
    (bh._core.axis.regular_numpy, (4, 2, 4), {}),
    (bh.axis.circular, (4, 2, 4), {}),
    (bh.axis.variable, ([1, 2, 3, 4],), {}),
    (bh.axis.integer, (1, 4), {}),
    (bh.axis.category, ([1, 2, 3],), {}),
    (bh.axis.category, ([1, 2, 3],), {"growth": True}),
    (bh.axis.category, (["1", "2", "3"],), {}),
    (bh.axis.category, (["1", "2", "3"],), {"growth": True}),
)


@pytest.mark.parametrize("axis,args,opts", axes_creations)
@pytest.mark.parametrize("copy_fn", copy_fns)
def test_axes(axis, args, opts, copy_fn):
    orig = axis(*args, metadata=None, **opts)
    new = copy_fn(orig)
    assert new == orig


@pytest.mark.parametrize("axis,args,opts", axes_creations)
@pytest.mark.parametrize("copy_fn", copy_fns)
def test_metadata_str(axis, args, opts, copy_fn):
    orig = axis(*args, metadata="foo", **opts)
    new = copy_fn(orig)
    assert new.metadata == orig.metadata
    new.metadata = orig.metadata
    assert new == orig


# Special test: Deepcopy should change metadata id, copy should not
@pytest.mark.parametrize("metadata", ({1: 2}, [1, 2, 3]))
def test_compare_copy_axis(metadata):
    orig = bh.axis.regular(4, 0, 2, metadata=metadata)
    new = copy.copy(orig)
    dnew = copy.deepcopy(orig)

    assert orig.metadata is new.metadata
    assert orig.metadata == dnew.metadata
    assert orig.metadata is not dnew.metadata


# Special test: Deepcopy should change metadata id, copy should not
@pytest.mark.parametrize("metadata", ({1: 2}, [1, 2, 3]))
def test_compare_copy_hist(metadata):
    orig = bh.histogram(bh.axis.regular(4, 0, 2, metadata=metadata))
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
    hist = bh.histogram(bh.axis.regular(4, 1, 2), bh.axis.regular(8, 3, 6))

    new = copy_fn(hist)
    assert hist == new


@pytest.mark.parametrize("copy_fn", copy_fns)
def test_histogram_fancy(copy_fn):
    hist = bh.histogram(bh.axis.regular(4, 1, 2), bh.axis.integer(0, 6))

    new = copy_fn(hist)
    assert hist == new


@pytest.mark.parametrize("copy_fn", copy_fns)
@pytest.mark.parametrize("metadata", ("This", (1, 2, 3)))
def test_histogram_metadata(copy_fn, metadata):

    hist = bh.histogram(bh.axis.regular(4, 1, 2, metadata=metadata))
    new = copy_fn(hist)
    assert hist == new


@pytest.mark.parametrize("copy_fn", copy_fns)
def test_numpy_edge(copy_fn):
    ax1 = bh._core.axis.regular_numpy(10, 0, 1)
    ax2 = copy_fn(ax1)

    # stop defaults to 0, so this fails if the copy fails
    assert ax1 == ax2
    assert ax1.index(1) == ax2.index(1)
    assert ax2.index(1) == 9
