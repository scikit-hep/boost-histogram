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
    assert h.axes[0] == integer(-1, 2)
    assert h.axes[0].extent == 5
    assert h.axes[0].size == 3
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
    assert h.axes[0].extent == 5

    with pytest.raises(IndexError):
        h[0, 1]

    for get in (lambda h, arg: h[arg], lambda h, arg: h[arg]):
        # lambda h, arg: h[arg]):
        assert get(h, 0) == 2
        assert get(h, 1) == 1
        assert get(h, 2) == 3
        # assert get(h, 0).variance == 2
        # assert get(h, 1).variance == 1
        # assert get(h, 2).variance == 3

    assert h[bh.overflow] == 1
    assert h[bh.underflow] == 1

    assert h[-1] == 3

    with pytest.raises(IndexError):
        h[3]
    with pytest.raises(IndexError):
        h[-3]


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
    assert h.axes[0].extent == 3 + 2 * flow

    with pytest.raises(IndexError):
        h[0, 1]

    for get in (lambda h, arg: h[arg],):
        # lambda h, arg: h[arg]):
        assert get(h, 0) == 2
        assert get(h, 1) == 1
        assert get(h, 2) == 3
        # assert get(h, 0).variance == 2
        # assert get(h, 1).variance == 1
        # assert get(h, 2).variance == 3

    if flow is True:
        assert get(h, bh.underflow) == 1
        assert get(h, bh.overflow) == 1


# TODO: atomic_int not supported
@pytest.mark.parametrize("storage", [bh.storage.int, bh.storage.double])
def test_setting(storage):
    h = histogram(regular(10, 0, 1), storage=storage)
    h[bh.underflow] = 1
    h[0] = 2
    h[1] = 3
    h[bh.loc(0.55)] = 4
    h[-1] = 5
    h[bh.overflow] = 6

    assert h[bh.underflow] == 1
    assert h[0] == 2
    assert h[1] == 3
    assert h[bh.loc(0.55)] == 4
    assert h[5] == 4
    assert h[-1] == 5
    assert h[9] == 5
    assert h[bh.overflow] == 6

    assert_array_equal(h.view(flow=True), [1, 2, 3, 0, 0, 0, 4, 0, 0, 0, 5, 6])


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
    assert h[bh.underflow] == 0
    assert h[0] == 1
    assert h[1] == 1000
    assert h[2] == 2
    assert h[bh.overflow] == 0


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

    for get in (lambda h, x, y: h._at(x, y),):
        # lambda h, x, y: h[x, y]):
        for i in range(-flow, h.axes[0].size + flow):
            for j in range(-flow, h.axes[1].size + flow):
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

    for i in range(-flow, h.axes[0].size + flow):
        for j in range(-flow, h.axes[1].size + flow):
            assert h._at(i, j) == 2 * m[i][j]


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

    for i in range(-flow, h.axes[0].size + flow):
        for j in range(-flow, h.axes[1].size + flow):
            assert h._at(i, j) == 2 * m[i][j]


def test_repr():
    hrepr = """histogram(
  regular(3, 0, 1),
  integer(0, 1),
  storage=double
)"""

    h = histogram(regular(3, 0, 1), integer(0, 1))
    assert repr(h) == hrepr

    h.fill([0.3, 0.5], [0, 0])
    hrepr += " # Sum: 2"
    assert repr(h) == hrepr

    h.fill([0.3, 12], [3, 0])
    hrepr += " (4 with flow)"
    assert repr(h) == hrepr


def test_axis():
    axes = (regular(10, 0, 1), integer(0, 1))
    h = histogram(*axes)
    for i, a in enumerate(axes):
        assert h.axes[i] == a
    with pytest.raises(IndexError):
        h.axes[2]
    assert h.axes[-1] == axes[-1]
    assert h.axes[-2] == axes[-2]
    with pytest.raises(IndexError):
        h.axes[-3]


def test_out_of_limit_axis():

    lim = bh.core.hist._axes_limit
    ax = (
        bh.axis.regular(1, -1, 1, underflow=False, overflow=False) for a in range(lim)
    )
    # Nothrow
    bh.histogram(*ax)

    ax = (
        bh.axis.regular(1, -1, 1, underflow=False, overflow=False)
        for a in range(lim + 1)
    )
    with pytest.raises(IndexError):
        bh.histogram(*ax)


def test_out_of_range():
    h = histogram(regular(3, 0, 1))
    h.fill(-1)
    h.fill(2)
    assert h[bh.underflow] == 1
    assert h[bh.overflow] == 1
    with pytest.raises(IndexError):
        h[4]


# CLASSIC: This used to have variance
def test_operators():
    h = histogram(integer(0, 2))
    h.fill(0)
    h += h
    assert h[0] == 2
    assert h[1] == 0
    h *= 2
    assert h[0] == 4
    assert h[1] == 0
    assert (h + h)[0] == (h * 2)[0]
    assert (h + h)[0] == (2 * h)[0]
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
    assert h0.axes[0] == integer(0, 2)
    assert [h0[i] for i in range(2)] == [2, 1]

    h1 = h.project(1)
    assert h1.rank == 1
    assert h1.axes[0] == integer(1, 4)
    assert [h1[i] for i in range(3)] == [1, 1, 1]

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
    for i in range(a.axes[0].extent):
        a.fill(i, 0, 0, 0, 0)
        for j in range(a.axes[1].extent):
            a.fill(i, j, 0, 0, 0)
            for k in range(a.axes[2].extent):
                a.fill(i, j, k, 0, 0)
                for l in range(a.axes[3].extent):
                    a.fill(i, j, k, l, 0)
                    for m in range(a.axes[4].extent):
                        a.fill(i, j, k, l, m * 0.5 * np.pi)

    io = pickle.dumps(a, -1)
    b = pickle.loads(io)

    assert id(a) != id(b)
    assert a.rank == b.rank
    assert a.axes[0] == b.axes[0]
    assert a.axes[1] == b.axes[1]
    assert a.axes[2] == b.axes[2]
    assert a.axes[3] == b.axes[3]
    assert a.axes[4] == b.axes[4]
    assert a.sum() == b.sum()
    assert repr(a) == repr(b)
    assert str(a) == str(b)
    assert a == b


def test_pickle_1():
    a = histogram(
        category([0, 1, 2]),
        integer(0, 3, metadata="ia"),
        regular(4, 0.0, 4.0, underflow=False, overflow=False),
        variable([0.0, 1.0, 2.0]),
    )
    assert isinstance(a, histogram)

    for i in range(a.axes[0].extent):
        a.fill(i, 0, 0, 0, weight=3)
        for j in range(a.axes[1].extent):
            a.fill(i, j, 0, 0, weight=10)
            for k in range(a.axes[2].extent):
                a.fill(i, j, k, 0, weight=2)
                for l in range(a.axes[3].extent):
                    a.fill(i, j, k, l, weight=5)

    io = BytesIO()
    pickle.dump(a, io, protocol=-1)
    io.seek(0)
    b = pickle.load(io)

    assert id(a) != id(b)
    assert a.rank == b.rank
    assert a.axes[0] == b.axes[0]
    assert a.axes[1] == b.axes[1]
    assert a.axes[2] == b.axes[2]
    assert a.axes[3] == b.axes[3]
    assert a.sum() == b.sum()
    assert repr(a) == repr(b)
    assert str(a) == str(b)
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
    for i in range(a.axes[0].extent):
        for j in range(a.axes[1].extent):
            for k in range(a.axes[2].extent):
                for m in range(i + j + k):
                    a.fill(i, j, k)
                r[i, j, k] = i + j + k

    d = np.zeros((2, 3, 4), dtype=np.int8)
    for i in range(a.axes[0].extent):
        for j in range(a.axes[1].extent):
            for k in range(a.axes[2].extent):
                d[i, j, k] = a[i, j, k]

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
    for i in range(a.axes[0].extent):
        for j in range(a.axes[1].extent):
            for k in range(a.axes[2].extent):
                a.fill(i - 1, j - 1, k - 1, weight=i + j + k)
                r[i, j, k] = i + j + k
    c = a.view(flow=True)

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
    assert a[0] == 2
    assert a[1] == 3
    assert a[2] == 2

    with pytest.raises(ValueError):
        a.fill(np.empty((2, 2)))
    with pytest.raises(ValueError):
        a.fill(np.empty(2), 1)
    with pytest.raises(ValueError):
        a.fill(np.empty(2), np.empty(3))
    with pytest.raises(ValueError):
        a.fill("abc")

    with pytest.raises(IndexError):
        a[1, 2]

    a = histogram(
        integer(0, 2, underflow=False, overflow=False),
        regular(2, 0, 2, underflow=False, overflow=False),
    )
    a.fill(ar(-1, 0, 1), ar(-1.0, 1.0, 0.1))
    assert a[0, 0] == 0
    assert a[0, 1] == 1
    assert a[1, 0] == 1
    assert a[1, 1] == 0
    # we don't support: assert a[[1, 1]].value, 0

    with pytest.raises(ValueError):
        a.fill(1)
    with pytest.raises(ValueError):
        a.fill([1, 0, 2], [1, 1])

    # This actually broadcasts
    a.fill([1, 0], [1])

    with pytest.raises(IndexError):
        a[1]
    with pytest.raises(IndexError):
        a[1, 2, 3]

    a = histogram(integer(0, 3, underflow=False, overflow=False))
    a.fill(ar(0, 0, 1, 2, 1, 0, 2, 2))
    assert a[0] == 3
    assert a[1] == 2
    assert a[2] == 3


def test_fill_with_numpy_array_1():
    def ar(*args):
        return np.array(args, dtype=float)

    a = histogram(integer(0, 3), storage=bh.storage.weight())
    v = ar(-1, 0, 1, 2, 3, 4)
    w = ar(2, 3, 4, 5, 6, 7)  # noqa
    a.fill(v, weight=w)
    a.fill((0, 1), weight=(2, 3))

    assert a[bh.underflow] == bh.accumulators.weighted_sum(2, 4)
    assert a[0] == bh.accumulators.weighted_sum(5, 13)
    assert a[1] == bh.accumulators.weighted_sum(7, 25)
    assert a[2] == bh.accumulators.weighted_sum(5, 25)

    assert a[bh.underflow].value == 2
    assert a[0].value == 5
    assert a[1].value == 7
    assert a[2].value == 5

    assert a[bh.underflow].variance == 4
    assert a[0].variance == 13
    assert a[1].variance == 25
    assert a[2].variance == 25

    a.fill((1, 2), weight=1)
    a.fill(0, weight=1)
    a.fill(0, weight=2)
    assert a[0].value == 8
    assert a[1].value == 8
    assert a[2].value == 6

    with pytest.raises(KeyError):
        a.fill((1, 2), foo=(1, 1))
    with pytest.raises(ValueError):
        a.fill((1, 2, 3), weight=(1, 2))
    with pytest.raises(ValueError):
        a.fill((1, 2), weight="ab")
    with pytest.raises(KeyError):
        a.fill((1, 2), weight=(1, 1), foo=1)
    with pytest.raises(ValueError):
        a.fill((1, 2), weight=([1, 1], [2, 2]))

    # CLASSIC: Used to fail
    a = histogram(integer(0, 3))
    a.fill((1, 2), weight=(1,))
    assert a[1] == 1.0
    assert a[2] == 1.0

    a = histogram(
        integer(0, 2, underflow=False, overflow=False),
        regular(2, 0, 2, underflow=False, overflow=False),
    )
    a.fill((-1, 0, 1), (-1, 1, 0.1))
    assert a[0, 0] == 0
    assert a[0, 1] == 1
    assert a[1, 0] == 1
    assert a[1, 1] == 0
    a = histogram(integer(0, 3, underflow=False, overflow=False))
    a.fill((0, 0, 1, 2))
    a.fill((1, 0, 2, 2))
    assert a[0] == 3
    assert a[1] == 2
    assert a[2] == 3


def test_fill_with_numpy_array_2():
    a = histogram(category(["A", "B"]))
    a.fill(("A", "B", "C"))
    a.fill(np.array(("D", "A"), dtype="S5"))
    assert a[0] == 2
    assert a[1] == 1
    assert a[bh.overflow] == 2

    b = histogram(integer(0, 2, underflow=False, overflow=False), category(["A", "B"]))
    b.fill((1, 0, 10), ("C", "B", "A"))
    assert b[0, 0] == 0
    assert b[1, 0] == 0
    assert b[0, 1] == 1
    assert b[1, 1] == 0
    assert b[0, bh.overflow] == 0
    assert b[1, bh.overflow] == 1
