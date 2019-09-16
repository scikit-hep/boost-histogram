import pytest

try:
    # Python 2
    from cPickle import loads, dumps
except ImportError:
    from pickle import loads, dumps

import copy

import boost.histogram as bh

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

copies = (copy.copy, copy.deepcopy)

storages = (
    bh.storage.atomic_int,
    bh.storage.double,
    bh.storage.int,
    bh.storage.profile,
    bh.storage.unlimited,
    bh.storage.weight,
    bh.storage.weighted_profile,
)


@pytest.mark.parametrize("accum,args", accumulators)
@pytest.mark.parametrize("copy_fn", copies)
def test_accumulators(accum, args, copy_fn):
    orig = accum(*args)
    new = copy_fn(orig)
    assert new == orig


axes_creations = (
    (bh.axis._regular_uoflow, (4, 2, 4)),
    (bh.axis._regular_growth, (4, 2, 4)),
    (bh.axis._regular_noflow, (4, 2, 4)),
    (bh.axis.regular_log, (4, 2, 4)),
    (bh.axis.regular_sqrt, (4, 2, 4)),
    (bh.axis.regular_pow, (4, 2, 4, 0.5)),
    (bh.axis.circular, (4, 2, 4)),
    (bh.axis.variable, ([1, 2, 3, 4],)),
    (bh.axis._integer_uoflow, (1, 4)),
    (bh.axis._category_int, ([1, 2, 3],)),
    (bh.axis._category_int_growth, ([1, 2, 3],)),
    (bh.axis._category_str, (["1", "2", "3"],)),
    (bh.axis._category_str_growth, (["1", "2", "3"],)),
)


@pytest.mark.parametrize("axis,args", axes_creations)
@pytest.mark.parametrize("copy_fn", copies)
def test_axes(axis, args, copy_fn):
    orig = axis(*args)
    new = copy_fn(orig)
    assert new == orig


@pytest.mark.parametrize("axis,args", axes_creations)
@pytest.mark.parametrize("copy_fn", copies)
def test_metadata_str(axis, args, copy_fn):
    orig = axis(*args, metadata="hi")
    new = copy_fn(orig)
    assert new.metadata == orig.metadata
    new.metadata = orig.metadata
    assert new == orig


# Special test: Deepcopy should change metadata id, copy should not
@pytest.mark.parametrize("metadata", ({1: 2}, [1, 2, 3]))
def test_compare_copy_axis(metadata):
    orig = bh.axis._regular_noflow(4, 0, 2, metadata=metadata)
    new = copy.copy(orig)
    dnew = copy.deepcopy(orig)

    assert orig.metadata is new.metadata
    assert orig.metadata == dnew.metadata
    assert orig.metadata is not dnew.metadata


# Special test: Deepcopy should change metadata id, copy should not
@pytest.mark.parametrize("metadata", ({1: 2}, [1, 2, 3]))
def test_compare_copy_hist(metadata):
    orig = bh._make_histogram(bh.axis._regular_noflow(4, 0, 2, metadata=metadata))
    new = copy.copy(orig)
    dnew = copy.deepcopy(orig)

    assert orig.axis(0).metadata is new.axis(0).metadata
    assert orig.axis(0).metadata == dnew.axis(0).metadata
    assert orig.axis(0).metadata is not dnew.axis(0).metadata


@pytest.mark.parametrize("axis,args", axes_creations)
@pytest.mark.parametrize("copy_fn", copies)
def test_metadata_any(axis, args, copy_fn):
    orig = axis(*args, metadata=(1, 2, 3))
    new = copy_fn(orig)
    assert new.metadata == orig.metadata
    new.metadata = orig.metadata
    assert new == orig


@pytest.mark.parametrize("copy_fn", copies)
@pytest.mark.parametrize("storage", storages)
def test_storage_int(copy_fn, storage):
    storage = storage()

    new = copy_fn(storage)
    assert storage == new


@pytest.mark.parametrize("copy_fn", copies)
def test_histogram_regular(copy_fn):
    hist = bh.histogram(bh.axis.regular(4, 1, 2), bh.axis.regular(8, 3, 6))

    new = copy_fn(hist)
    assert hist == new


@pytest.mark.parametrize("copy_fn", copies)
def test_histogram_fancy(copy_fn):
    hist = bh.histogram(bh.axis._regular_noflow(4, 1, 2), bh.axis._integer_uoflow(0, 6))

    new = copy_fn(hist)
    assert hist == new


@pytest.mark.parametrize("copy_fn", copies)
@pytest.mark.parametrize("metadata", ("This", (1, 2, 3)))
def test_histogram_metadata(copy_fn, metadata):

    hist = bh.histogram(bh.axis.regular(4, 1, 2, metadata=metadata))
    new = copy_fn(hist)
    assert hist == new
