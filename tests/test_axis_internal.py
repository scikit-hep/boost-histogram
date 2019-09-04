import pytest
from pytest import approx

import boost.histogram as bh
import numpy as np
from numpy.testing import assert_array_equal


@pytest.mark.parametrize("axtype", [bh.axis._regular_uoflow, bh.axis._regular_uflow, bh.axis._regular_oflow, bh.axis._regular_noflow])
@pytest.mark.parametrize("function", [lambda x: x,
                                       lambda x: bh._make_histogram(x).axis(0),
                                       ])
def test_axis__regular_uoflow(axtype, function):
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

    for b, v in zip(ax.bins(), np.arange(10)/10.0):
        assert b.lower() == approx(v)
        assert b.upper() == approx(v + .1)

def test_axis_regular_extents():
    ax = bh.axis._regular_uoflow(10,0,1)
    assert 12 == ax.size(flow=True)
    assert 11 == len(ax.edges())
    assert 13 == len(ax.edges(True))
    assert 10 == len(ax.centers())
    assert ax.options() == bh.axis.options.underflow | bh.axis.options.overflow

    ax = bh.axis._regular_uflow(10,0,1)
    assert 11 == ax.size(flow=True)
    assert 11 == len(ax.edges())
    assert 12 == len(ax.edges(True))
    assert 10 == len(ax.centers())
    assert ax.options() == bh.axis.options.underflow

    ax = bh.axis._regular_oflow(10,0,1)
    assert 11 == ax.size(flow=True)
    assert 11 == len(ax.edges())
    assert 12 == len(ax.edges(True))
    assert 10 == len(ax.centers())
    assert ax.options() == bh.axis.options.overflow

    ax = bh.axis._regular_noflow(10,0,1)
    assert 10 == ax.size(flow=True)
    assert 11 == len(ax.edges())
    assert 11 == len(ax.edges(True))
    assert 10 == len(ax.centers())
    assert ax.options() == bh.axis.options.none

def test_axis_growth():
    ax = bh.axis._regular_growth(10,0,1)
    ax.index(.7)
    ax.index(1.2)
    assert ax.size() == 10
    assert len(ax.centers()) == 10
    assert len(ax.edges()) == 11
    assert ax.update(1.21) == (12,-3)
    assert ax.size() == 13
    assert len(ax.edges()) == 14
    assert len(ax.centers()) == 13

def test_axis_growth_cat():
    ax = bh.axis._category_str_growth(["This"])
    assert ax.size() == 1
    ax.update("That")
    assert ax.size() == 2
    assert ax.bin(0) == "This"
    assert ax.bin(1) == "That"

@pytest.mark.parametrize("offset", [-1,0,1,2])
def test_axis_circular_offset(offset):
    ax = bh.axis.circular(10, 0, 1)
    assert 11 == len(ax.edges())

    assert 3 == ax.index(.34 + offset)
    assert 2 == ax.index(.26 + offset)

    assert_array_equal([2,3], ax.index([.26 + offset, .34 + offset]))

def test_axis_circular():
    ax = bh.axis.circular(10, 0, 1)

    assert .1 == ax.value(1)
    assert ax.options() == bh.axis.options.circular | bh.axis.options.overflow

normal_axs = [
    bh.axis._regular_uoflow,
    bh.axis._regular_noflow,
    bh.axis.circular,
    bh.axis.regular_log,
    bh.axis.regular_sqrt,
]

@pytest.mark.parametrize("axis", normal_axs)
def test_regular_axis_repr(axis):
    ax = axis(2,3,4)
    assert 'object at' not in repr(ax)

    ax = axis(7,2,4, metadata='This')
    assert 'This' in repr(ax)
    assert ax.metadata == 'This'

    ax.metadata = 'That'
    ax = axis(7,2,4, metadata='That')
    assert 'That' in repr(ax)
    assert ax.metadata == 'That'

def test_metadata_compare():
    ax1 = bh.axis._regular_uoflow(1,2,3, metadata=[1,])
    ax2 = bh.axis._regular_uoflow(1,2,3, metadata=[1,])

    assert ax1 == ax2

def test_metadata_compare_neq():
    ax1 = bh.axis._regular_uoflow(1,2,3, metadata=[1,])
    ax2 = bh.axis._regular_uoflow(1,2,3, metadata=[2,])

    assert ax1 != ax2

@pytest.mark.parametrize("axis", normal_axs)
def test_any_metadata(axis):
    ax = axis(2,3,4, metadata={"one": "1"})
    assert ax.metadata == {"one": "1"}
    ax.metadata = 64
    assert ax.metadata == 64

def test_cat_str():
    ax = bh.axis._category_str(["a", "b", "c"])
    assert ax.bin(0) == "a"
    assert ax.bin(1) == "b"
    assert ax.bin(2) == "c"

    assert ax.index("b") == 1

def test_cat_int():
    ax = bh.axis._category_int([1,2,3])
    assert ax.bin(0) == 1
    assert ax.bin(1) == 2
    assert ax.bin(2) == 3

    assert ax.index(2) == 1
