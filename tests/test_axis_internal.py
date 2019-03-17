import pytest

import histogram as bh
import numpy as np


@pytest.mark.parametrize("axtype", [bh.axis.regular, bh.axis.regular_noflow])
def test_axis_regular(axtype):
    ax = axtype(10, 0, 1)

    assert 3 == ax.index(.34)
    assert 2 == ax.index(.26)
    assert -1 == ax.index(-.23)
    assert -1 == ax.index(-5.23)
    assert 10 == ax.index(1.01)
    assert 10 == ax.index(23)

    assert 10 == ax.size()

def test_axis_regular_extents():
    ax = bh.axis.regular(10,0,1)
    assert 12 == ax.extent()
    assert ax.options() == bh.axis.options.underflow | bh.axis.options.overflow

    ax = bh.axis.regular_noflow(10,0,1)
    assert 10 == ax.extent()
    assert ax.options() == bh.axis.options.none

@pytest.mark.parametrize("offset", [-1,0,1,2])
def test_axis_circular_offset(offset):
    ax = bh.axis.circular(10, 0, 1)

    assert 3 == ax.index(.34 + offset)
    assert 2 == ax.index(.26 + offset)

def test_axis_circular():
    ax = bh.axis.circular(10, 0, 1)

    assert .1 == ax.value(1)
    assert ax.options() == bh.axis.options.circular | bh.axis.options.overflow


@pytest.mark.parametrize("axis", [
    bh.axis.regular,
    bh.axis.regular_noflow,
    bh.axis.circular,
    bh.axis.regular_log,
    bh.axis.regular_sqrt,
])
def test_regular_axis_repr(axis):
    ax = axis(2,3,4)
    assert 'object at' not in repr(ax)
