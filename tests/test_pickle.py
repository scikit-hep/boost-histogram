import pytest

try:
    # Python 2
    from cPickle import loads, dumps
except ImportError:
    from pickle import loads, dumps

import boost.histogram as bh


class TestAccumulators:
    def test_sum(self):
        orig = bh.accumulators.sum(12)
        new = loads(dumps(orig))
        assert new == orig

    def test_weighted_sum(self):
        orig = bh.accumulators.weighted_sum(1.5, 2.5)
        new = loads(dumps(orig))
        assert new == orig

    def test_mean(self):
        orig = bh.accumulators.mean(5, 1.5, 2.5)
        new = loads(dumps(orig))
        assert new == orig

    def test_weighted_mean(self):
        orig = bh.accumulators.weighted_mean(1.5, 2.5, 3.5, 4.5)
        new = loads(dumps(orig))
        assert new == orig


axes_creations = (
        (bh.axis.regular_uoflow,      (4, 2, 4)),
        (bh.axis.regular_growth,      (4, 2, 4)),
        (bh.axis.regular_noflow,      (4, 2, 4)),
        (bh.axis.regular_log,         (4, 2, 4)),
        (bh.axis.regular_sqrt,        (4, 2, 4)),
        (bh.axis.regular_pow,         (4, 2, 4, 0.5)),
        (bh.axis.circular,            (4, 2, 4)),
        (bh.axis.variable,            ([1, 2, 3, 4],)),
        (bh.axis.integer_uoflow,      (1, 4)),
        (bh.axis.category_int,        ([1, 2, 3],)),
        (bh.axis.category_int_growth, ([1, 2, 3],)),
        (bh.axis.category_str,        (["1", "2", "3"],)),
        (bh.axis.category_str_growth, (["1", "2", "3"],)),
        )

@pytest.mark.parametrize("axis,args", axes_creations)
def test_axes(axis, args):
    orig = axis(*args)
    new = loads(dumps(orig))
    assert new == orig


@pytest.mark.parametrize("axis,args", axes_creations)
def test_metadata_str(axis, args):
    orig = axis(*args, metadata="hi")
    new = loads(dumps(orig))
    assert new.metadata == orig.metadata
    new.metadata = orig.metadata
    assert new == orig

@pytest.mark.parametrize("axis,args", axes_creations)
def test_metadata_any(axis, args):
    orig = axis(*args, metadata=(1,2,3))
    new = loads(dumps(orig))
    assert new.metadata == orig.metadata
    new.metadata = orig.metadata
    assert new == orig


def test_storage_int():
    storage = bh.storage.int()
    storage.push_back(1)
    storage.push_back(3)
    storage.push_back(2)

    assert storage[0] == 1
    assert storage[1] == 3
    assert storage[2] == 2

    new = loads(dumps(storage))
    assert storage == new

def test_histogram_regular():
    hist = bh.histogram(bh.axis.regular(4,1,2), bh.axis.regular(8,3,6))

    new = loads(dumps(hist))
    assert hist == new


def test_histogram_fancy():
    hist = bh.histogram(bh.axis.regular_noflow(4,1,2), bh.axis.integer_uoflow(0, 6))

    new = loads(dumps(hist))
    assert hist == new

def test_histogram_metadata():

    hist = bh.histogram(bh.axis.regular(4,1,2, metadata="This"))
    new = loads(dumps(hist))
    assert hist.axis(0).metadata == new.axis(0).metadata

    hist = bh.histogram(bh.axis.regular(4,1,2, metadata=(1,2,3)))
    new = loads(dumps(hist))
    assert hist.axis(0).metadata == new.axis(0).metadata

    # Note that == directly will not work since it is "is" in Python
