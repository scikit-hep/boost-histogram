import pytest
from pytest import approx


from boost.histogram import make_histogram as histogram
from boost.histogram.axis import (regular, regular_noflow,
                                  regular_log, regular_sqrt,
                                  regular_pow, circular,
                                  variable, integer,
                                  category_int as category)

import numpy as np

import abc

# compatible with Python 2 *and* 3:
ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})

# histogram -> boost.histogram
# histogram -> make_histogram
# dim -> rank()
# ax.shape -> ax.extent()

def test_init():
    histogram()
    histogram(integer(-1, 1))
    with pytest.raises(RuntimeError):
        histogram(1)
    with pytest.raises(RuntimeError):
        histogram("bla")
    with pytest.raises(RuntimeError):
        histogram([])
    with pytest.raises(RuntimeError):
        histogram(regular)
    with pytest.raises(TypeError):
        histogram(regular())
    with pytest.raises(RuntimeError):
        histogram([integer(-1, 1)])
     # TODO: Should fail
     # CLASSIC: with pytest.raises(ValueError):
    histogram(integer(-1, 1), unknown_keyword="nh")

    h = histogram(integer(-1, 2))
    assert h.rank() == 1
    assert h.axis(0) == integer(-1, 2)
    assert h.axis(0).extent() == 5
    assert h.axis(0).size() == 3
    assert h != histogram(regular(1, -1, 1))
    assert h != histogram(integer(-1, 1, metadata="ia"))

