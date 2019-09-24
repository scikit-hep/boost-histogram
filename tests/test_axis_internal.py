import pytest
from pytest import approx

import boost_histogram as bh
import numpy as np
from numpy.testing import assert_array_equal


@pytest.mark.parametrize(
    "axtype",
    [
        bh.core.axis._regular_uoflow,
        bh.core.axis._regular_uflow,
        bh.core.axis._regular_oflow,
        bh.core.axis._regular_noflow,
    ],
)
@pytest.mark.parametrize("function", [lambda x: x, lambda x: bh.histogram(x).axis(0)])
def test_axis_regular_uoflow(axtype, function):
    ax = function(axtype(10, 0, 1))

    assert 3 == ax.index(0.34)
    assert 2 == ax.index(0.26)
    assert -1 == ax.index(-0.23)
    assert -1 == ax.index(-5.23)
    assert 10 == ax.index(1.01)
    assert 10 == ax.index(23)

    assert 10 == ax.size

    assert ax.bin(3)[0] == approx(0.3)
    assert ax.bin(3)[1] == approx(0.4)

    for b, v in zip(ax, np.arange(10) / 10.0):
        assert b[0] == approx(v)
        assert b[1] == approx(v + 0.1)


def test_axis_regular_extents():
    ax = bh.axis.regular(10, 0, 1)
    assert 12 == ax.extent
    assert 11 == len(ax.edges())
    assert 13 == len(ax.edges(True))
    assert 10 == len(ax.centers())
    assert ax.options == bh.axis.options(underflow=True, overflow=True)

    ax = bh.axis.regular(10, 0, 1, overflow=False)
    assert 11 == ax.extent
    assert 11 == len(ax.edges())
    assert 12 == len(ax.edges(True))
    assert 10 == len(ax.centers())
    assert ax.options == bh.axis.options(underflow=True)

    ax = bh.axis.regular(10, 0, 1, underflow=False)
    assert 11 == ax.extent
    assert 11 == len(ax.edges())
    assert 12 == len(ax.edges(True))
    assert 10 == len(ax.centers())
    assert ax.options == bh.axis.options(overflow=True)

    ax = bh.axis.regular(10, 0, 1, flow=False)
    assert 10 == ax.extent
    assert 11 == len(ax.edges())
    assert 11 == len(ax.edges(True))
    assert 10 == len(ax.centers())
    assert ax.options == bh.axis.options()


def test_axis_growth():
    ax = bh.axis.regular(10, 0, 1, growth=True)
    ax.index(0.7)
    ax.index(1.2)
    assert ax.size == 10
    assert len(ax.centers()) == 10
    assert len(ax.edges()) == 11
    assert ax.update(1.21) == (12, -3)
    assert ax.size == 13
    assert len(ax.edges()) == 14
    assert len(ax.centers()) == 13


def test_axis_growth_cat():
    ax = bh.axis.category(["This"], growth=True)
    assert ax.size == 1
    ax.update("That")
    assert ax.size == 2
    assert ax.bin(0) == "This"
    assert ax.bin(1) == "That"


@pytest.mark.parametrize("offset", [-1, 0, 1, 2])
def test_axis_circular_offset(offset):
    ax = bh.axis.circular(10, 0, 1)
    assert 11 == len(ax.edges())

    assert 3 == ax.index(0.34 + offset)
    assert 2 == ax.index(0.26 + offset)

    assert_array_equal([2, 3], ax.index([0.26 + offset, 0.34 + offset]))


def test_axis_circular():
    ax = bh.axis.circular(10, 0, 1)

    assert 0.1 == ax.value(1)
    assert ax.options == bh.axis.options(circular=True, overflow=True)


normal_axs = [
    bh.core.axis._regular_uoflow,
    bh.core.axis._regular_noflow,
    bh.core.axis.circular,
    bh.core.axis.regular_log,
    bh.core.axis.regular_sqrt,
]


@pytest.mark.parametrize("axis", normal_axs)
def test_regular_axis_repr(axis):
    ax = axis(2, 3, 4)
    assert "object at" not in repr(ax)

    ax = axis(7, 2, 4, metadata="This")
    assert "This" in repr(ax)
    assert ax.metadata == "This"

    ax.metadata = "That"
    ax = axis(7, 2, 4, metadata="That")
    assert "That" in repr(ax)
    assert ax.metadata == "That"


def test_metadata_compare():
    ax1 = bh.axis.regular(1, 2, 3, metadata=[1])
    ax2 = bh.axis.regular(1, 2, 3, metadata=[1])

    assert ax1 == ax2


def test_metadata_compare_neq():
    ax1 = bh.axis.regular(1, 2, 3, metadata=[1])
    ax2 = bh.axis.regular(1, 2, 3, metadata=[2])

    assert ax1 != ax2


@pytest.mark.parametrize("axis", normal_axs)
def test_any_metadata(axis):
    ax = axis(2, 3, 4, metadata={"one": "1"})
    assert ax.metadata == {"one": "1"}
    ax.metadata = 64
    assert ax.metadata == 64


def test_cat_str():
    ax = bh.axis.category(["a", "b", "c"])
    assert ax.bin(0) == "a"
    assert ax.bin(1) == "b"
    assert ax.bin(2) == "c"

    assert ax.index("b") == 1
    assert_array_equal(ax.index(("b", "a", "f")), [1, 0, 3])
    # assert ax.value(0) == "a"
    assert_array_equal(ax.value((1, 2)), ("b", "c"))


def test_cat_int():
    ax = bh.axis.category([1, 2, 3])
    assert ax.bin(0) == 1
    assert ax.bin(1) == 2
    assert ax.bin(2) == 3

    assert ax.index(2) == 1
    assert_array_equal(ax.index((2, 1, 5)), [1, 0, 3])
