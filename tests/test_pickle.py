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
    (bh.axis.Regular, (4, 2, 4), {"transform": bh.axis.transform.Log()}),
    (bh.axis.Regular, (4, 2, 4), {"transform": bh.axis.transform.Sqrt()}),
    (bh.axis.Regular, (4, 2, 4), {"transform": bh.axis.transform.Pow(0.5)}),
    (bh._core.axis.regular_numpy, (4, 2, 4), {}),
    (bh.axis.Regular, (4, 2, 4), {"circular": True}),
    (bh.axis.Variable, ([1, 2, 3, 4],), {}),
    (bh.axis.Integer, (1, 4), {}),
    (bh.axis.Category, ([1, 2, 3],), {}),
    (bh.axis.Category, ([1, 2, 3],), {"growth": True}),
    (bh.axis.Category, (["1", "2", "3"],), {}),
    (bh.axis.Category, (["1", "2", "3"],), {"growth": True}),
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
    ax1 = bh._core.axis.regular_numpy(10, 0, 1)
    ax2 = copy_fn(ax1)

    # stop defaults to 0, so this fails if the copy fails
    assert ax1 == ax2
    assert ax1.index(1) == ax2.index(1)
    assert ax2.index(1) == 9


# CLASSIC: This used to have metadata too, but that does not compare equal
def test_pickle_0():
    a = bh.Histogram(
        bh.axis.Category([0, 1, 2]),
        bh.axis.Integer(0, 20),
        bh.axis.Regular(2, 0.0, 20.0, underflow=False, overflow=False),
        bh.axis.Variable([0.0, 1.0, 2.0]),
        bh.axis.Regular(4, 0, 2 * np.pi, circular=True),
    )
    for i in range(a.axes[0].extent):
        a.fill(i, 0, 0, 0, 0)
        for j in range(a.axes[1].extent):
            a.fill(i, j, 0, 0, 0)
            for k in range(a.axes[2].extent):
                a.fill(i, j, k, 0, 0)
                for l in range(a.axes[3].extent):
                    a.fill(i, j, k, l, 0)
                    for m in range(a.axes[4].extent):
                        a.fill(i, j, k, l, m * 0.5 * np.pi)

    io = pickle.dumps(a, -1)
    b = pickle.loads(io)

    assert id(a) != id(b)
    assert a.rank == b.rank
    assert a.axes[0] == b.axes[0]
    assert a.axes[1] == b.axes[1]
    assert a.axes[2] == b.axes[2]
    assert a.axes[3] == b.axes[3]
    assert a.axes[4] == b.axes[4]
    assert a.sum() == b.sum()
    assert repr(a) == repr(b)
    assert str(a) == str(b)
    assert a == b


def test_pickle_1():
    a = bh.Histogram(
        bh.axis.Category([0, 1, 2]),
        bh.axis.Integer(0, 3, metadata="ia"),
        bh.axis.Regular(4, 0.0, 4.0, underflow=False, overflow=False),
        bh.axis.Variable([0.0, 1.0, 2.0]),
    )
    assert isinstance(a, bh.Histogram)

    for i in range(a.axes[0].extent):
        a.fill(i, 0, 0, 0, weight=3)
        for j in range(a.axes[1].extent):
            a.fill(i, j, 0, 0, weight=10)
            for k in range(a.axes[2].extent):
                a.fill(i, j, k, 0, weight=2)
                for l in range(a.axes[3].extent):
                    a.fill(i, j, k, l, weight=5)

    io = BytesIO()
    pickle.dump(a, io, protocol=-1)
    io.seek(0)
    b = pickle.load(io)

    assert id(a) != id(b)
    assert a.rank == b.rank
    assert a.axes[0] == b.axes[0]
    assert a.axes[1] == b.axes[1]
    assert a.axes[2] == b.axes[2]
    assert a.axes[3] == b.axes[3]
    assert a.sum() == b.sum()
    assert repr(a) == repr(b)
    assert str(a) == str(b)
    assert a == b
