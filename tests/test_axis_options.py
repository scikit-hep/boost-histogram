# -*- coding: utf-8 -*-
from boost_histogram.axis import Traits


def test_compare():
    assert Traits() == Traits()
    assert Traits(underflow=True) == Traits(underflow=True)
    assert Traits(overflow=True) == Traits(overflow=True)
    assert Traits(circular=True) == Traits(circular=True)
    assert Traits(growth=True) == Traits(growth=True)
    assert Traits(underflow=True, overflow=True) == Traits(
        underflow=True, overflow=True
    )
    assert Traits() != Traits(overflow=True)
    assert Traits(underflow=True) != Traits(overflow=True)
    assert Traits(underflow=True, overflow=True) != Traits(overflow=True)


def test_fields():
    o = Traits()

    assert not o.underflow
    assert not o.overflow
    assert not o.circular
    assert not o.growth

    o = Traits(underflow=True, overflow=True)

    assert o.underflow
    assert o.overflow
    assert not o.circular
    assert not o.growth
