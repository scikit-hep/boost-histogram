import pytest

try:
    # Python 2
    from cPickle import loads, dumps
except ImportError:
    from pickle import loads, dumps

import boost.histogram as bh

class TestAccumulator:
    def test_sum(self):
        orig = bh.accumulator.sum(12)
        new = loads(dumps(orig))
        assert new == orig

    def test_weighted_sum(self):
        orig = bh.accumulator.weighted_sum(1.5, 2.5)
        new = loads(dumps(orig))
        assert new == orig

    def test_mean(self):
        orig = bh.accumulator.mean(5, 1.5, 2.5)
        new = loads(dumps(orig))
        assert new == orig

    def test_weighted_mean(self):
        orig = bh.accumulator.weighted_mean(1.5, 2.5, 3.5, 4.5)
        new = loads(dumps(orig))
        assert new == orig


axes_creations = (
        (bh.axis.regular,             (4, 2, 4)),
        (bh.axis.regular_growth,      (4, 2, 4)),
        (bh.axis.regular_noflow,      (4, 2, 4)),
        (bh.axis.regular_log,         (4, 2, 4)),
        (bh.axis.regular_sqrt,        (4, 2, 4)),
        (bh.axis.regular_pow,         (0.5, 4, 2, 4)),
        (bh.axis.circular,            (4, 2, 4)),
        (bh.axis.variable,            ([1, 2, 3, 4],)),
        (bh.axis.integer,             (1, 4)),
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


