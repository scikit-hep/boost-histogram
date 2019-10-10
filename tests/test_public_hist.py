import pytest
from pytest import approx


from boost_histogram import histogram
from boost_histogram.axis import (
    regular,
    integer,
    regular_log,
    regular_sqrt,
    regular_pow,
    circular,
    variable,
    category,
)

import boost_histogram as bh

import numpy as np
from numpy.testing import assert_array_equal
from io import BytesIO

try:
    import cPickle as pickle
except ImportError:
    import pickle

# histogram -> boost_histogram
# histogram -> histogram
# .dim -> .rank


def test_init():
    histogram()
    histogram(integer(-1, 1))
    with pytest.raises(TypeError):
        histogram(1)
    with pytest.raises(TypeError):
        histogram("bla")
    with pytest.raises(TypeError):
        histogram([])
    with pytest.raises(TypeError):
        histogram(regular)
    with pytest.raises(TypeError):
        histogram(regular())
    with pytest.raises(TypeError):
        histogram([integer(-1, 1)])
    with pytest.raises(TypeError):
        histogram([integer(-1, 1), integer(-1, 1)])
    with pytest.raises(TypeError):
        histogram(integer(-1, 1), unknown_keyword="nh")

    h = histogram(integer(-1, 2))
    assert h.rank == 1
    assert h.axis(0) == integer(-1, 2)
    assert h.axis(0).extent == 5
    assert h.axis(0).size == 3
    assert h != histogram(regular(1, -1, 1))
    assert h != histogram(integer(-1, 1, metadata="ia"))


def test_copy():
    a = histogram(integer(-1, 1))
    import copy

    b = copy.copy(a)
    assert a == b
    assert id(a) != id(b)

    c = copy.deepcopy(b)
    assert b == c
    assert id(b) != id(c)


def test_fill_int_1d():

    h = histogram(integer(-1, 2))
    assert isinstance(h, histogram)

    with pytest.raises(ValueError):
        h.fill()
    with pytest.raises(ValueError):
        h.fill(1, 2)
    for x in (-10, -1, -1, 0, 1, 1, 1, 10):
        h.fill(x)
    assert h.sum() == 6
    assert h.sum(flow=True) == 8
    assert h.axis(0).extent == 5

    with pytest.raises(TypeError):
        h.at(0, foo=None)
    with pytest.raises(ValueError):
        h.at(0, 1)
    with pytest.raises(IndexError):
        h[0, 1]

    for get in (lambda h, arg: h.at(arg),):
        # lambda h, arg: h[arg]):
        assert get(h, 0) == 2
        assert get(h, 1) == 1
        assert get(h, 2) == 3
        # assert get(h, 0).variance == 2
        # assert get(h, 1).variance == 1
        # assert get(h, 2).variance == 3

        assert get(h, -1) == 1
        assert get(h, 3) == 1


@pytest.mark.parametrize("flow", [True, False])
def test_fill_1d(flow):
    h = histogram(regular(3, -1, 2, underflow=flow, overflow=flow))
    with pytest.raises(ValueError):
        h.fill()
    with pytest.raises(ValueError):
        h.fill(1, 2)
    for x in (-10, -1, -1, 0, 1, 1, 1, 10):
        h.fill(x)

    assert h.sum() == 6
    assert h.sum(flow=True) == 6 + 2 * flow
    assert h.axis(0).extent == 3 + 2 * flow

    with pytest.raises(TypeError):
        h.at(0, foo=None)
    with pytest.raises(ValueError):
        h.at(0, 1)
    with pytest.raises(IndexError):
        h[0, 1]

    for get in (lambda h, arg: h.at(arg),):
        # lambda h, arg: h[arg]):
        assert get(h, 0) == 2
        assert get(h, 1) == 1
        assert get(h, 2) == 3
        # assert get(h, 0).variance == 2
        # assert get(h, 1).variance == 1
        # assert get(h, 2).variance == 3

    if flow is True:
        assert get(h, -1) == 1
        assert get(h, 3) == 1


def test_growth():
    h = histogram(integer(-1, 2))
    h.fill(-1)
    h.fill(1)
    h.fill(1)
    for i in range(255):
        h.fill(0)
    h.fill(0)
    for i in range(1000 - 256):
        h.fill(0)
    assert h.at(-1) == 0
    assert h.at(0) == 1
    assert h.at(1) == 1000
    assert h.at(2) == 2
    assert h.at(3) == 0


@pytest.mark.parametrize("flow", [True, False])
def test_fill_2d(flow):
    h = histogram(
        integer(-1, 2, underflow=flow, overflow=flow),
        regular(4, -2, 2, underflow=flow, overflow=flow),
    )
    h.fill(-1, -2)
    h.fill(-1, -1)
    h.fill(0, 0)
    h.fill(0, 1)
    h.fill(1, 0)
    h.fill(3, -1)
    h.fill(0, -3)

    with pytest.raises(Exception):
        h.fill(1)
    with pytest.raises(Exception):
        h.fill(1, 2, 3)

    m = [
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    for get in (lambda h, x, y: h.at(x, y),):
        # lambda h, x, y: h[x, y]):
        for i in range(-flow, h.axis(0).size + flow):
            for j in range(-flow, h.axis(1).size + flow):
                assert get(h, i, j) == m[i][j]


@pytest.mark.parametrize("flow", [True, False])
def test_add_2d(flow):
    h = histogram(
        integer(-1, 2, underflow=flow, overflow=flow),
        regular(4, -2, 2, underflow=flow, overflow=flow),
    )
    assert isinstance(h, histogram)

    h.fill(-1, -2)
    h.fill(-1, -1)
    h.fill(0, 0)
    h.fill(0, 1)
    h.fill(1, 0)
    h.fill(3, -1)
    h.fill(0, -3)

    m = [
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    h += h

    for i in range(-flow, h.axis(0).size + flow):
        for j in range(-flow, h.axis(1).size + flow):
            assert h.at(i, j) == 2 * m[i][j]


def test_add_2d_bad():
    a = histogram(integer(-1, 1))
    b = histogram(regular(3, -1, 1))

    with pytest.raises(ValueError):
        a += b


@pytest.mark.parametrize("flow", [True, False])
def test_add_2d_w(flow):
    h = histogram(
        integer(-1, 2, underflow=flow, overflow=flow),
        regular(4, -2, 2, underflow=flow, overflow=flow),
    )
    h.fill(-1, -2)
    h.fill(-1, -1)
    h.fill(0, 0)
    h.fill(0, 1)
    h.fill(1, 0)
    h.fill(3, -1)
    h.fill(0, -3)

    m = [
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    h2 = histogram(
        integer(-1, 2, underflow=flow, overflow=flow),
        regular(4, -2, 2, underflow=flow, overflow=flow),
    )
    h2.fill(0, 0, weight=0)

    h2 += h
    h2 += h
    h += h
    assert h == h2

    for i in range(-flow, h.axis(0).size + flow):
        for j in range(-flow, h.axis(1).size + flow):
            assert h.at(i, j) == 2 * m[i][j]


def test_repr():
    h = histogram(regular(3, 0, 1), integer(0, 1))
    hr = repr(h)
    assert (
        hr
        == """histogram(
  regular(3, 0, 1),
  integer(0, 1),
  storage=double
)"""
    )


def test_axis():
    axes = (regular(10, 0, 1), integer(0, 1))
    h = histogram(*axes)
    for i, a in enumerate(axes):
        assert h.axis(i) == a
    with pytest.raises(IndexError):
        h.axis(2)
    assert h.axis(-1) == axes[-1]
    assert h.axis(-2) == axes[-2]
    with pytest.raises(IndexError):
        h.axis(-3)


# CLASSIC: This used to only fail when accessing, now fails in creation
def test_overflow():
    with pytest.raises(RuntimeError):
        h = histogram(*[regular(1, 0, 1) for i in range(50)])


def test_out_of_range():
    h = histogram(regular(3, 0, 1))
    h.fill(-1)
    h.fill(2)
    assert h.at(-1) == 1
    assert h.at(3) == 1
    with pytest.raises(IndexError):
        h.at(-2)
    with pytest.raises(IndexError):
        h.at(4)
    # with pytest.raises(IndexError):
    #    h.at(-2).variance
    # with pytest.raises(IndexError):
    #    h.at(4).variance


# CLASSIC: This used to have variance
def test_operators():
    h = histogram(integer(0, 2))
    h.fill(0)
    h += h
    assert h.at(0) == 2
    assert h.at(1) == 0
    h *= 2
    assert h.at(0) == 4
    assert h.at(1) == 0
    assert (h + h).at(0) == (h * 2).at(0)
    assert (h + h).at(0) == (2 * h).at(0)
    h2 = histogram(regular(2, 0, 2))
    with pytest.raises(ValueError):
        h + h2


# CLASSIC: reduce_to -> project,
def test_project():
    h = histogram(integer(0, 2), integer(1, 4))
    h.fill(0, 1)
    h.fill(0, 2)
    h.fill(1, 3)

    h0 = h.project(0)
    assert h0.rank == 1
    assert h0.axis(0) == integer(0, 2)
    assert [h0.at(i) for i in range(2)] == [2, 1]

    h1 = h.project(1)
    assert h1.rank == 1
    assert h1.axis(0) == integer(1, 4)
    assert [h1.at(i) for i in range(3)] == [1, 1, 1]

    with pytest.raises(ValueError):
        h.project(*range(10))

    with pytest.raises(ValueError):
        h.project(2, 1)


def test_shrink_1d_external_reduce():
    h = histogram(regular(20, 1, 5))
    h.fill(1.1)
    hs = bh.algorithm.reduce(h, bh.algorithm.shrink(0, 1, 2))
    assert_array_equal(hs.view(), [1, 0, 0, 0, 0])


def test_shrink_1d():
    h = histogram(regular(20, 1, 5))
    h.fill(1.1)
    hs = h.reduce(bh.algorithm.shrink(0, 1, 2))
    assert_array_equal(hs.view(), [1, 0, 0, 0, 0])


def test_rebin_1d():
    h = histogram(regular(20, 1, 5))
    h.fill(1.1)
    hs = h.reduce(bh.algorithm.rebin(0, 4))
    assert_array_equal(hs.view(), [1, 0, 0, 0, 0])


def test_shrink_rebin_1d():
    h = histogram(regular(20, 0, 4))
    h.fill(1.1)
    hs = h.reduce(bh.algorithm.shrink_and_rebin(0, 1, 3, 2))
    assert_array_equal(hs.view(), [1, 0, 0, 0, 0])


# CLASSIC: This used to have metadata too, but that does not compare equal
def test_pickle_0():
    a = histogram(
        category([0, 1, 2]),
        integer(0, 20),
        regular(2, 0.0, 20.0, underflow=False, overflow=False),
        variable([0.0, 1.0, 2.0]),
        circular(4, 0, 2 * np.pi),
    )
    for i in range(a.axis(0).extent):
        a.fill(i, 0, 0, 0, 0)
        for j in range(a.axis(1).extent):
            a.fill(i, j, 0, 0, 0)
            for k in range(a.axis(2).extent):
                a.fill(i, j, k, 0, 0)
                for l in range(a.axis(3).extent):
                    a.fill(i, j, k, l, 0)
                    for m in range(a.axis(4).extent):
                        a.fill(i, j, k, l, m * 0.5 * np.pi)

    io = pickle.dumps(a, -1)
    b = pickle.loads(io)

    assert id(a) != id(b)
    assert a.rank == b.rank
    assert a.axis(0) == b.axis(0)
    assert a.axis(1) == b.axis(1)
    assert a.axis(2) == b.axis(2)
    assert a.axis(3) == b.axis(3)
    assert a.axis(4) == b.axis(4)
    assert a.sum() == b.sum()
    assert a == b


def test_pickle_1():
    a = histogram(
        category([0, 1, 2]),
        integer(0, 3, metadata="ia"),
        regular(4, 0.0, 4.0, underflow=False, overflow=False),
        variable([0.0, 1.0, 2.0]),
    )
    assert isinstance(a, histogram)

    for i in range(a.axis(0).extent):
        a.fill(i, 0, 0, 0, weight=3)
        for j in range(a.axis(1).extent):
            a.fill(i, j, 0, 0, weight=10)
            for k in range(a.axis(2).extent):
                a.fill(i, j, k, 0, weight=2)
                for l in range(a.axis(3).extent):
                    a.fill(i, j, k, l, weight=5)

    io = BytesIO()
    pickle.dump(a, io, protocol=-1)
    io.seek(0)
    b = pickle.load(io)

    assert id(a) != id(b)
    assert a.rank == b.rank
    assert a.axis(0) == b.axis(0)
    assert a.axis(1) == b.axis(1)
    assert a.axis(2) == b.axis(2)
    assert a.axis(3) == b.axis(3)
    assert a.sum() == b.sum()
    assert a == b


# Numpy tests


def test_numpy_conversion_0():
    a = histogram(integer(0, 3, underflow=False, overflow=False))
    a.fill(0)
    for i in range(5):
        a.fill(1)
    c = np.array(a)  # a copy
    v = np.asarray(a)  # a view

    for t in (c, v):
        assert t.dtype == np.double  # CLASSIC: np.uint8
        assert_array_equal(t, (1, 5, 0))

    for i in range(10):
        a.fill(2)
    # copy does not change, but view does
    assert_array_equal(c, (1, 5, 0))
    assert_array_equal(v, (1, 5, 10))

    for i in range(255):
        a.fill(1)
    c = np.array(a)

    assert c.dtype == np.double  # CLASSIC: np.uint16
    assert_array_equal(c, (1, 260, 10))
    # view does not follow underlying switch in word size
    # assert not np.all(c, v)


def test_numpy_conversion_1():
    # CLASSIC: was weight array
    a = histogram(integer(0, 3))
    for i in range(10):
        a.fill(1, weight=3)
    c = np.array(a)  # a copy
    v = np.asarray(a)  # a view
    assert c.dtype == np.double  # CLASSIC: np.float64
    assert_array_equal(c, np.array((0, 30, 0)))
    assert_array_equal(v, c)


def test_numpy_conversion_2():
    a = histogram(
        integer(0, 2, underflow=False, overflow=False),
        integer(0, 3, underflow=False, overflow=False),
        integer(0, 4, underflow=False, overflow=False),
    )
    r = np.zeros((2, 3, 4), dtype=np.int8)
    for i in range(a.axis(0).extent):
        for j in range(a.axis(1).extent):
            for k in range(a.axis(2).extent):
                for m in range(i + j + k):
                    a.fill(i, j, k)
                r[i, j, k] = i + j + k

    d = np.zeros((2, 3, 4), dtype=np.int8)
    for i in range(a.axis(0).extent):
        for j in range(a.axis(1).extent):
            for k in range(a.axis(2).extent):
                d[i, j, k] = a.at(i, j, k)

    assert_array_equal(d, r)

    c = np.array(a)  # a copy
    v = np.asarray(a)  # a view

    assert_array_equal(c, r)
    assert_array_equal(v, r)


def test_numpy_conversion_3():
    # It's okay to forget the () on a storage
    a = histogram(
        integer(0, 2), integer(0, 3), integer(0, 4), storage=bh.storage.double
    )

    r = np.zeros((4, 5, 6))
    for i in range(a.axis(0).extent):
        for j in range(a.axis(1).extent):
            for k in range(a.axis(2).extent):
                a.fill(i - 1, j - 1, k - 1, weight=i + j + k)
                r[i, j, k] = i + j + k
    c = a.view(flow=True)

    c2 = np.zeros((4, 5, 6))
    for i in range(a.axis(0).extent):
        for j in range(a.axis(1).extent):
            for k in range(a.axis(2).extent):
                c2[i, j, k] = a.at(i - 1, j - 1, k - 1)

    assert_array_equal(c, c2)
    assert_array_equal(c, r)

    assert a.sum() == approx(144)
    assert a.sum(flow=True) == approx(720)
    assert c.sum() == approx(720)


def test_numpy_conversion_4():
    a = histogram(
        integer(0, 2, underflow=False, overflow=False),
        integer(0, 4, underflow=False, overflow=False),
    )
    a1 = np.asarray(a)
    assert a1.dtype == np.double
    assert a1.shape == (2, 4)

    b = histogram()
    b1 = np.asarray(b)
    assert b1.shape == ()
    assert np.sum(b1) == 0

    # Compare sum methods
    assert b.sum() == np.asarray(b).sum()


def test_numpy_conversion_5():
    a = histogram(
        integer(0, 3, underflow=False, overflow=False),
        integer(0, 2, underflow=False, overflow=False),
        storage=bh.storage.unlimited(),
    )
    a.fill(0, 0)
    for i in range(80):
        a = a + a
    # a now holds a multiprecision type
    a.fill(1, 0)
    for i in range(2):
        a.fill(2, 0)
    for i in range(3):
        a.fill(0, 1)
    for i in range(4):
        a.fill(1, 1)
    for i in range(5):
        a.fill(2, 1)
    a1 = a.view()
    assert a1.shape == (3, 2)
    assert a1[0, 0] == float(2 ** 80)
    assert a1[1, 0] == 1
    assert a1[2, 0] == 2
    assert a1[0, 1] == 3
    assert a1[1, 1] == 4
    assert a1[2, 1] == 5


def test_fill_with_numpy_array_0():
    def ar(*args):
        return np.array(args, dtype=float)

    a = histogram(integer(0, 3, underflow=False, overflow=False))
    a.fill(ar(-1, 0, 1, 2, 1))
    a.fill((4, -1, 0, 1, 2))
    assert a.at(0) == 2
    assert a.at(1) == 3
    assert a.at(2) == 2

    with pytest.raises(ValueError):
        a.fill(np.empty((2, 2)))
    with pytest.raises(ValueError):
        a.fill(np.empty(2), 1)
    with pytest.raises(ValueError):
        a.fill(np.empty(2), np.empty(3))
    with pytest.raises(ValueError):
        a.fill("abc")

    with pytest.raises(ValueError):
        a.at(1, 2)

    a = histogram(
        integer(0, 2, underflow=False, overflow=False),
        regular(2, 0, 2, underflow=False, overflow=False),
    )
    a.fill(ar(-1, 0, 1), ar(-1.0, 1.0, 0.1))
    assert a.at(0, 0) == 0
    assert a.at(0, 1) == 1
    assert a.at(1, 0) == 1
    assert a.at(1, 1) == 0
    # we don't support: assert a.at([1, 1]).value, 0

    with pytest.raises(ValueError):
        a.fill(1)
    with pytest.raises(ValueError):
        a.fill([1, 0, 2], [1, 1])

    # This actually broadcasts
    a.fill([1, 0], [1])

    with pytest.raises(ValueError):
        a.at(1)
    with pytest.raises(ValueError):
        a.at(1, 2, 3)

    a = histogram(integer(0, 3, underflow=False, overflow=False))
    a.fill(ar(0, 0, 1, 2, 1, 0, 2, 2))
    assert a.at(0) == 3
    assert a.at(1) == 2
    assert a.at(2) == 3


def test_fill_with_numpy_array_1():
    def ar(*args):
        return np.array(args, dtype=float)

    a = histogram(integer(0, 3), storage=bh.storage.weight())
    v = ar(-1, 0, 1, 2, 3, 4)
    w = ar(2, 3, 4, 5, 6, 7)  # noqa
    a.fill(v, weight=w)
    a.fill((0, 1), weight=(2, 3))

    assert a.at(-1) == bh.accumulators.weighted_sum(2, 4)
    assert a.at(0) == bh.accumulators.weighted_sum(5, 13)
    assert a.at(1) == bh.accumulators.weighted_sum(7, 25)
    assert a.at(2) == bh.accumulators.weighted_sum(5, 25)

    assert a.at(-1).value == 2
    assert a.at(0).value == 5
    assert a.at(1).value == 7
    assert a.at(2).value == 5

    assert a.at(-1).variance == 4
    assert a.at(0).variance == 13
    assert a.at(1).variance == 25
    assert a.at(2).variance == 25

    a.fill((1, 2), weight=1)
    a.fill(0, weight=1)
    a.fill(0, weight=2)
    assert a.at(0).value == 8
    assert a.at(1).value == 8
    assert a.at(2).value == 6

    with pytest.raises(KeyError):
        a.fill((1, 2), foo=(1, 1))
    with pytest.raises(ValueError):
        a.fill((1, 2), weight=(1,))
    with pytest.raises(ValueError):
        a.fill((1, 2), weight="ab")
    with pytest.raises(KeyError):
        a.fill((1, 2), weight=(1, 1), foo=1)
    with pytest.raises(ValueError):
        a.fill((1, 2), weight=([1, 1], [2, 2]))

    a = histogram(
        integer(0, 2, underflow=False, overflow=False),
        regular(2, 0, 2, underflow=False, overflow=False),
    )
    a.fill((-1, 0, 1), (-1, 1, 0.1))
    assert a.at(0, 0) == 0
    assert a.at(0, 1) == 1
    assert a.at(1, 0) == 1
    assert a.at(1, 1) == 0
    a = histogram(integer(0, 3, underflow=False, overflow=False))
    a.fill((0, 0, 1, 2))
    a.fill((1, 0, 2, 2))
    assert a.at(0) == 3
    assert a.at(1) == 2
    assert a.at(2) == 3


def test_fill_with_numpy_array_2():
    a = histogram(category(["A", "B"]))
    a.fill(("A", "B", "C"))
    a.fill(np.array(("D", "A"), dtype="S5"))
    assert a.at(0) == 2
    assert a.at(1) == 1
    assert a.at(2) == 2

    b = histogram(integer(0, 2, underflow=False, overflow=False), category(["A", "B"]))
    b.fill((1, 0, 10), ("C", "B", "A"))
    assert b.at(0, 0) == 0
    assert b.at(1, 0) == 0
    assert b.at(0, 1) == 1
    assert b.at(1, 1) == 0
    assert b.at(0, 2) == 0
    assert b.at(1, 2) == 1
