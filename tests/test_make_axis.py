import pytest

import boost.histogram as bh

def test_make_regular_normal():
    ax_reg = bh.axis.regular(10, 0, 1)
    assert isinstance(ax_reg, bh.axis.regular_uoflow)
    assert isinstance(ax_reg, bh.axis.regular)

def test_make_regular_uoflow():
    ax_reg = bh.axis.regular(10, 0, 1, flow=True)
    assert isinstance(ax_reg, bh.axis.regular_uoflow)
    assert isinstance(ax_reg, bh.axis.regular)

def test_make_regular_noflow():
    ax_reg = bh.axis.regular(10, 0, 1, flow=False)
    assert isinstance(ax_reg, bh.axis.regular_noflow)
    assert isinstance(ax_reg, bh.axis.regular)

def test_make_regular_growth():
    ax_reg = bh.axis.regular(10, 0, 1, growth=True)
    assert isinstance(ax_reg, bh.axis.regular_growth)
    assert isinstance(ax_reg, bh.axis.regular)
