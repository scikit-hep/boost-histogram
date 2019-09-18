import pytest

import boost_histogram as bh


def test_make_regular_normal():
    ax_reg = bh.axis.regular(10, 0, 1)
    assert isinstance(ax_reg, bh.core.axis._regular_uoflow)
    assert isinstance(ax_reg, bh.axis.regular)


def test_make__regular_uoflow():
    ax_reg = bh.axis.regular(10, 0, 1, flow=True)
    assert isinstance(ax_reg, bh.core.axis._regular_uoflow)
    assert isinstance(ax_reg, bh.axis.regular)


def test_make__regular_noflow():
    ax_reg = bh.axis.regular(10, 0, 1, flow=False)
    assert isinstance(ax_reg, bh.core.axis._regular_noflow)
    assert isinstance(ax_reg, bh.axis.regular)


def test_make__regular_growth():
    ax_reg = bh.axis.regular(10, 0, 1, growth=True)
    assert isinstance(ax_reg, bh.core.axis._regular_growth)
    assert isinstance(ax_reg, bh.axis.regular)


def test_make__category_int():
    ax = bh.axis.category([1, 2, 3])
    assert isinstance(ax, bh.core.axis._category_int)
    assert isinstance(ax, bh.axis.category)


def test_make__category_int_growth():
    ax = bh.axis.category([1, 2, 3], growth=True)
    assert isinstance(ax, bh.core.axis._category_int_growth)
    assert isinstance(ax, bh.axis.category)


def test_make__category_str():
    ax = bh.axis.category(["one", "two"])
    assert isinstance(ax, bh.core.axis._category_str)
    assert isinstance(ax, bh.axis.category)


def test_make__category_str_growth():
    ax = bh.axis.category(["one", "two"], growth=True)
    assert isinstance(ax, bh.core.axis._category_str_growth)
    assert isinstance(ax, bh.axis.category)
