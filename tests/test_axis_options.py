# -*- coding: utf-8 -*-
from boost_histogram.axis import options


def test_compare():
    assert options() == options()
    assert options(underflow=True) == options(underflow=True)
    assert options(overflow=True) == options(overflow=True)
    assert options(circular=True) == options(circular=True)
    assert options(growth=True) == options(growth=True)
    assert options(underflow=True, overflow=True) == options(
        underflow=True, overflow=True
    )
    assert options() != options(overflow=True)
    assert options(underflow=True) != options(overflow=True)
    assert options(underflow=True, overflow=True) != options(overflow=True)


def test_fields():
    o = options()

    assert not o.underflow
    assert not o.overflow
    assert not o.circular
    assert not o.growth

    o = options(underflow=True, overflow=True)

    assert o.underflow
    assert o.overflow
    assert not o.circular
    assert not o.growth
