import pytest
from pytest import approx

import histogram as bh
import numpy as np


@pytest.mark.parametrize("axtype", [bh.axis.regular, bh.axis.regular_noflow])
@pytest.mark.parametrize("function", [lambda x: x,
                                       lambda x: bh.make_histogram(x).axis(0),
                                       ])
def test_axis_regular(axtype, function):
    ax = function(axtype(10, 0, 1))

    assert 3 == ax.index(.34)
    assert 2 == ax.index(.26)
    assert -1 == ax.index(-.23)
    assert -1 == ax.index(-5.23)
    assert 10 == ax.index(1.01)
    assert 10 == ax.index(23)

    assert 10 == ax.size()

    assert ax.bin(3).lower() == approx(.3)
    assert ax.bin(3).upper() == approx(.4)
    assert ax.bin(3).width() == approx(.1)
    assert ax.bin(3).center() == approx(.35)

def test_axis_regular_extents():
    ax = bh.axis.regular(10,0,1)
    assert 12 == ax.extent()
    assert ax.options() == bh.axis.options.underflow | bh.axis.options.overflow

    ax = bh.axis.regular_noflow(10,0,1)
    assert 10 == ax.extent()
    assert ax.options() == bh.axis.options.none

def test_axis_growth():
    ax = bh.axis.regular_growth(10,0,1)
    ax.index(.7)
    ax.index(1.2)
    assert ax.size() == 10
    assert ax.update(1.2) == (12,-3)
    assert ax.size() == 13

def test_axis_growth_cat():
    ax = bh.axis.category_str_growth(["This"])
    assert ax.size() == 1
    ax.update("That")
    assert ax.size() == 2
    assert ax.bin(0) == "This"
    assert ax.bin(1) == "That"

@pytest.mark.parametrize("offset", [-1,0,1,2])
def test_axis_circular_offset(offset):
    ax = bh.axis.circular(10, 0, 1)

    assert 3 == ax.index(.34 + offset)
    assert 2 == ax.index(.26 + offset)

    assert np.all([2,3] == ax.index([.26 + offset, .34 + offset]))

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

    ax = axis(7,2,4, label='This')
    assert 'This' in repr(ax)
    assert ax.label == 'This'

    ax.label = 'That'
    ax = axis(7,2,4, label='That')
    assert 'That' in repr(ax)
    assert ax.label == 'That'

def test_cat_str():
    ax = bh.axis.category_str(["a", "b", "c"])
    assert ax.bin(0) == "a"
    assert ax.bin(1) == "b"
    assert ax.bin(2) == "c"
