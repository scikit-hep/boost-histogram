import boost.histogram as bh

import pytest

def test_1D_get_bin():

    h = bh.histogram(bh.axis.regular(10, 0, 1))
    h.fill([.25, .25, .25, .15])

    assert h[0] == 0
    assert h[1] == 1
    assert h[2] == 3

    assert h.view()[2] == h[2]

    with pytest.raises(IndexError):
        h[1,2]

def test_2D_get_bin():

    h = bh.histogram(bh.axis.regular(10, 0, 1), bh.axis.regular(10, 0, 1))
    h.fill(.15, [.25, .25, .25, .15])

    assert h[0,0] == 0
    assert h[0,1] == 0
    assert h[1,1] == 1
    assert h[1,2] == 3

    assert h.view()[1,2] == h[1,2]

    with pytest.raises(IndexError):
        h[1]


def test_get_1D_histogram():
    h = bh.histogram(bh.axis.regular(10, 0, 1))
    h.fill([.25, .25, .25, .15])

    h2 = h[:]

    assert h == h2

    h.fill(.75)

    assert h != h2

