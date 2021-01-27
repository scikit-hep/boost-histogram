# -*- coding: utf-8 -*-
import abc
import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx

import boost_histogram as bh
import boost_histogram.utils

# compatible with Python 2 *and* 3:
ABC = abc.ABCMeta("ABC", (object,), {"__slots__": ()})


@pytest.mark.parametrize(
    "axis,args,opt,kwargs",
    [
        (bh.axis.Regular, (1, 2, 3), "", {}),
        (bh.axis.Regular, (1, 2, 3), "u", {}),
        (bh.axis.Regular, (1, 2, 3), "o", {}),
        (bh.axis.Regular, (1, 2, 3), "uo", {}),
        (bh.axis.Regular, (1, 2, 3), "g", {}),
        (bh.axis.Regular, (1, 2, 3), "", {"circular": True}),
        (bh.axis.Regular, (1, 2, 3), "", {"transform": bh.axis.transform.log}),
        (bh.axis.Regular, (1, 2, 3), "", {"transform": bh.axis.transform.sqrt}),
        (bh.axis.Regular, (1, 2, 3), "", {"transform": bh.axis.transform.Pow(1)}),
        (bh.axis.Variable, ((1, 2, 3),), "", {}),
        (bh.axis.Variable, ((1, 2, 3),), "u", {}),
        (bh.axis.Variable, ((1, 2, 3),), "o", {}),
        (bh.axis.Variable, ((1, 2, 3),), "uo", {}),
        (bh.axis.Variable, ((1, 2, 3),), "g", {}),
        (bh.axis.Variable, ((1, 2, 3),), "", {"circular": True}),
        (bh.axis.Integer, (1, 2), "", {}),
        (bh.axis.Integer, (1, 2), "u", {}),
        (bh.axis.Integer, (1, 2), "o", {}),
        (bh.axis.Integer, (1, 2), "uo", {}),
        (bh.axis.Integer, (1, 2), "g", {}),
        (bh.axis.Integer, (1, 2), "", {"circular": True}),
        (bh.axis.IntCategory, ((1, 2, 3),), "", {}),
        (bh.axis.IntCategory, ((1, 2, 3),), "g", {}),
        (bh.axis.StrCategory, (tuple("ABC"),), "", {}),
        (bh.axis.StrCategory, (tuple("ABC"),), "g", {}),
    ],
)
def test_metadata(axis, args, opt, kwargs):
    for m in ("foo", 64, {"one": 1}):
        if "u" in opt:
            kwargs["underflow"] = False
        if "o" in opt:
            kwargs["overflow"] = False
        if "g" in opt:
            kwargs["growth"] = True
        kwargs["metadata"] = m

        assert axis(*args, **kwargs).metadata is m
        mcopy = copy.deepcopy(m)
        assert axis(*args, **kwargs).metadata == m
        assert axis(*args, **kwargs).metadata == mcopy
        assert axis(*args, **kwargs).metadata != "bar"
        assert axis(*args, **kwargs) == axis(*args, **kwargs)
        assert axis(*args, **kwargs) != axis(*args, metadata="bar")

    del kwargs["metadata"]

    ax = axis(*args, __dict__={"metadata": 3, "other": 2})
    assert ax.metadata == 3
    assert ax.other == 2

    del ax.__dict__
    assert ax.__dict__ == {}
    assert ax.metadata is None

    ax.__dict__ = {"metadata": 5}
    assert ax.__dict__ == {"metadata": 5}
    assert ax.metadata == 5

    # Python 2 does not allow mixing ** and kw
    new_kwargs = copy.copy(kwargs)
    new_kwargs["__dict__"] = {"something": 2}
    new_kwargs["metadata"] = 3
    with pytest.raises(KeyError):
        axis(*args, **new_kwargs)

    new_kwargs = copy.copy(kwargs)
    new_kwargs["__dict__"] = {"metadata": 2}
    new_kwargs["metadata"] = 3
    with pytest.raises(KeyError):
        axis(*args, **new_kwargs)


# The point of this ABC is to force all the tests listed here to be implemented
# for each axis type. PyTest instantiates these test classes for us, so missing
# one really does fail the test.
class Axis(ABC):
    @abc.abstractmethod
    def test_init(self):
        pass

    @abc.abstractmethod
    def test_traits(self):
        pass

    @abc.abstractmethod
    def test_equal(self):
        pass

    @abc.abstractmethod
    def test_len(self):
        pass

    @abc.abstractmethod
    def test_repr(self):
        pass

    @abc.abstractmethod
    def test_getitem(self):
        pass

    @abc.abstractmethod
    def test_iter(self):
        pass

    @abc.abstractmethod
    def test_index(self):
        pass

    @abc.abstractmethod
    def test_edges_centers_widths(self):
        pass


class TestRegular(Axis):
    def test_init(self):
        # Should not throw
        bh.axis.Regular(1, 1.0, 2.0)
        bh.axis.Regular(1, 1.0, 2.0, metadata="ra")
        bh.axis.Regular(1, 1.0, 2.0, underflow=False)
        bh.axis.Regular(1, 1.0, 2.0, underflow=False, overflow=False, metadata="ra")
        bh.axis.Regular(1, 1.0, 2.0, metadata=0)
        bh.axis.Regular(1, 1.0, 2.0, transform=bh.axis.transform.log)
        bh.axis.Regular(1, 1.0, 2.0, transform=bh.axis.transform.sqrt)
        bh.axis.Regular(1, 1.0, 2.0, transform=bh.axis.transform.Pow(1.5))

        with pytest.raises(TypeError):
            bh.axis.Regular()
        with pytest.raises(TypeError):
            bh.axis.Regular(overflow=False, underflow=False)
        with pytest.raises(TypeError):
            bh.axis.Regular(1)
        with pytest.raises(TypeError):
            bh.axis.Regular(1, 1.0)
        with pytest.raises(ValueError):
            bh.axis.Regular(0, 1.0, 2.0)
        with pytest.raises(TypeError):
            bh.axis.Regular("1", 1.0, 2.0)
        with pytest.raises(Exception):
            bh.axis.Regular(-1, 1.0, 2.0)

        with pytest.raises(ValueError):
            bh.axis.Regular(1, 1.0, 1.0)

        with pytest.raises(TypeError):
            bh.axis.Regular(1, 1.0, 2.0, bad_keyword="ra")
        with pytest.raises(AttributeError):
            bh.axis.Regular(1, 1.0, 2.0, transform=lambda x: 2)
        with pytest.raises(TypeError):
            bh.axis.Regular(1, 1.0, 2.0, transform=bh.axis.transform.Pow)
        # TODO: These errors could be better

    def test_traits(self):
        STD_TRAITS = dict(continuous=True, ordered=True)

        ax = bh.axis.Regular(1, 2, 3)
        assert isinstance(ax, bh.axis.Regular)
        assert ax.traits == bh.axis.Traits(underflow=True, overflow=True, **STD_TRAITS)

        ax = bh.axis.Regular(1, 2, 3, overflow=False)
        assert isinstance(ax, bh.axis.Regular)
        assert ax.traits == bh.axis.Traits(underflow=True, **STD_TRAITS)

        ax = bh.axis.Regular(1, 2, 3, underflow=False)
        assert isinstance(ax, bh.axis.Regular)
        assert ax.traits == bh.axis.Traits(overflow=True, **STD_TRAITS)

        ax = bh.axis.Regular(1, 2, 3, underflow=False, overflow=False)
        assert isinstance(ax, bh.axis.Regular)
        assert ax.traits == bh.axis.Traits(**STD_TRAITS)

        ax = bh.axis.Regular(1, 2, 3, growth=True)
        assert isinstance(ax, bh.axis.Regular)
        assert ax.traits == bh.axis.Traits(
            underflow=True, overflow=True, growth=True, **STD_TRAITS
        )

    def test_equal(self):
        a = bh.axis.Regular(4, 1.0, 2.0)
        assert a == bh.axis.Regular(4, 1.0, 2.0)
        assert a != bh.axis.Regular(3, 1.0, 2.0)
        assert a != bh.axis.Regular(4, 1.1, 2.0)
        assert a != bh.axis.Regular(4, 1.0, 2.1)

        # metadata compare
        assert bh.axis.Regular(1, 2, 3, metadata=1) == bh.axis.Regular(
            1, 2, 3, metadata=1
        )
        assert bh.axis.Regular(1, 2, 3, metadata=1) != bh.axis.Regular(
            1, 2, 3, metadata="1"
        )
        assert bh.axis.Regular(1, 2, 3, metadata=1) != bh.axis.Regular(
            1, 2, 3, metadata=[1]
        )

    def test_len(self):
        a = bh.axis.Regular(4, 1.0, 2.0)
        assert len(a) == 4
        assert a.size == 4
        assert a.extent == 6

    def test_repr(self):
        ax = bh.axis.Regular(4, 1.1, 2.2)
        assert repr(ax) == "Regular(4, 1.1, 2.2)"

        ax = bh.axis.Regular(4, 1.1, 2.2, metadata="ra")
        assert repr(ax) == "Regular(4, 1.1, 2.2)"

        ax = bh.axis.Regular(4, 1.1, 2.2, underflow=False)
        assert repr(ax) == "Regular(4, 1.1, 2.2, underflow=False)"

        ax = bh.axis.Regular(4, 1.1, 2.2, metadata="ra", overflow=False)
        assert repr(ax) == "Regular(4, 1.1, 2.2, overflow=False)"

        ax = bh.axis.Regular(4, 1.1, 2.2, metadata="ra", circular=True)
        assert repr(ax) == "Regular(4, 1.1, 2.2, circular=True)"

        ax = bh.axis.Regular(4, 1.1, 2.2, transform=bh.axis.transform.log)
        assert repr(ax) == "Regular(4, 1.1, 2.2, transform=log)"
        # TODO: Add caching so that an extracted functional transform actually works

        ax = bh.axis.Regular(3, 1.1, 2.2, transform=bh.axis.transform.sqrt)
        assert repr(ax) == "Regular(3, 1.1, 2.2, transform=sqrt)"

        ax = bh.axis.Regular(4, 1.1, 2.2, transform=bh.axis.transform.Pow(0.5))
        assert repr(ax) == "Regular(4, 1.1, 2.2, transform=pow(0.5))"

    def test_getitem(self):
        a = bh.axis.Regular(2, 1.0, 2.0)
        ref = [1.0, 1.5, 2.0]
        for i in range(2):
            assert_allclose(a.bin(i), ref[i : i + 2])
            assert_allclose(a[i], ref[i : i + 2])

        assert a[-1] == a[1]
        with pytest.raises(IndexError):
            a[2]

        assert a.bin(-1)[0] == -np.inf
        assert a.bin(2)[1] == np.inf

        assert_allclose(a[bh.underflow], a.bin(-1))
        assert_allclose(a[bh.overflow], a.bin(2))

        with pytest.raises(IndexError):
            a.bin(-2)
        with pytest.raises(IndexError):
            a.bin(3)

    def test_iter(self):
        a = bh.axis.Regular(2, 1.0, 2.0)
        ref = [(1.0, 1.5), (1.5, 2.0)]
        assert_allclose(a, ref)

    def test_index(self):
        a = bh.axis.Regular(4, 1.0, 2.0)

        assert a.index(-1) == -1
        assert a.index(0.99) == -1
        assert a.index(1.0) == 0
        assert a.index(1.249) == 0
        assert a.index(1.250) == 1
        assert a.index(1.499) == 1
        assert a.index(1.500) == 2
        assert a.index(1.749) == 2
        assert a.index(1.750) == 3
        assert a.index(1.999) == 3
        assert a.index(2.000) == 4
        assert a.index(20) == 4

    def test_reversed_index(self):
        a = bh.axis.Regular(4, 2.0, 1.0)

        assert a.index(-1) == 4
        assert a.index(0.99) == 4
        assert a.index(1.0) == 4
        assert a.index(1.249) == 3
        assert a.index(1.250) == 3
        assert a.index(1.499) == 2
        assert a.index(1.500) == 2
        assert a.index(1.749) == 1
        assert a.index(1.750) == 1
        assert a.index(1.999) == 0
        assert a.index(2.000) == 0
        assert a.index(20) == -1

    def test_sqrt_transform(self):
        a = bh.axis.Regular(10, 0, 10, transform=bh.axis.transform.sqrt)
        # Edges: 0. ,  0.1,  0.4,  0.9,  1.6,  2.5,  3.6,  4.9,  6.4,  8.1, 10.

        assert a.index(-100) == 10  # Always in overflow bin
        assert a.index(-1) == 10  # When transform is invalid
        assert a.index(0) == 0
        assert a.index(0.15) == 1
        assert a.index(0.5) == 2
        assert a.index(1) == 3
        assert a.index(1.7) == 4
        assert a.index(9) == 9
        assert a.index(11) == 10
        assert a.index(1000) == 10

        assert a.bin(0)[0] == approx(0)
        assert a.bin(1)[0] == approx(0.1)
        assert a.bin(1)[1] == approx(0.4)
        assert a.bin(2)[0] == approx(0.4)

    def test_log_transform(self):
        a = bh.axis.Regular(2, 1e0, 1e2, transform=bh.axis.transform.log)

        assert a.index(-1) == 2
        assert a.index(0.99) == -1
        assert a.index(1.0) == 0
        assert a.index(9.99) == 0
        assert a.index(10.0) == 1
        assert a.index(99.9) == 1
        assert a.index(100) == 2
        assert a.index(1000) == 2

        assert a.bin(0)[0] == approx(1e0)
        assert a.bin(1)[0] == approx(1e1)
        assert a.bin(1)[1] == approx(1e2)

    def test_pow_transform(self):
        a = bh.axis.Regular(2, 1.0, 9.0, transform=bh.axis.transform.Pow(0.5))

        assert a.index(-1) == 2
        assert a.index(0.99) == -1
        assert a.index(1.0) == 0
        assert a.index(3.99) == 0
        assert a.index(4.0) == 1
        assert a.index(8.99) == 1
        assert a.index(9) == 2
        assert a.index(1000) == 2

        assert a.bin(0)[0] == approx(1.0)
        assert a.bin(1)[0] == approx(4.0)
        assert a.bin(1)[1] == approx(9.0)

    def test_edges_centers_widths(self):
        a = bh.axis.Regular(2, 0, 1)
        assert_allclose(a.edges, [0, 0.5, 1])
        assert_allclose(a.centers, [0.25, 0.75])
        assert_allclose(a.widths, [0.5, 0.5])


class TestCircular(Axis):
    def test_init(self):
        # Should not throw
        bh.axis.Regular(1, 2, 3, circular=True)
        bh.axis.Regular(1, 2, 3, metadata="pa", circular=True)

        with pytest.raises(TypeError):
            bh.axis.Regular(1, 2, 3, "pa", circular=True)

        with pytest.raises(TypeError):
            bh.axis.Regular(circular=True)
        with pytest.raises(TypeError):
            bh.axis.Regular(1, circular=True)
        with pytest.raises(TypeError):
            bh.axis.Regular(1, -1, circular=True)
        with pytest.raises(Exception):
            bh.axis.Regular(-1, circular=True)
        with pytest.raises(TypeError):
            bh.axis.Regular(1, 1.0, metadata=1, circular=True)
        with pytest.raises(TypeError):
            bh.axis.Regular("1", circular=True)

    def test_traits(self):
        ax = bh.axis.Regular(1, 2, 3, circular=True)
        assert isinstance(ax, bh.axis.Regular)
        assert ax.traits == bh.axis.Traits(
            overflow=True, circular=True, continuous=True, ordered=True
        )

    def test_equal(self):
        a = bh.axis.Regular(4, 0.0, 1.0, circular=True)
        assert a == bh.axis.Regular(4, 0, 1, circular=True)
        assert a != bh.axis.Regular(2, 0, 1, circular=True)
        assert isinstance(a, bh.axis.Regular)

    def test_len(self):
        assert len(bh.axis.Regular(4, 0.0, 1.0, circular=True)) == 4
        assert bh.axis.Regular(4, 0.0, 1.0, circular=True).size == 4
        assert bh.axis.Regular(4, 0.0, 1.0, circular=True).extent == 5

    def test_repr(self):
        ax = bh.axis.Regular(4, 1.1, 2.2, circular=True)
        assert repr(ax) == "Regular(4, 1.1, 2.2, circular=True)"

        ax = bh.axis.Regular(4, 1.1, 2.2, metadata="hi", circular=True)
        assert repr(ax) == "Regular(4, 1.1, 2.2, circular=True)"

    def test_getitem(self):
        a = bh.axis.Regular(2, 1, 1 + np.pi * 2, circular=True)
        ref = [1.0, 1.0 + np.pi, 1.0 + 2.0 * np.pi]
        for i in range(2):
            assert_allclose(a.bin(i), ref[i : i + 2])
            assert_allclose(a[i], ref[i : i + 2])

        assert a[-1] == a[1]
        with pytest.raises(IndexError):
            a[2]

        with pytest.raises(IndexError):
            a[bh.underflow]

        assert_allclose(a[bh.overflow], a.bin(2))

        assert a.bin(2)[0] == approx(1 + 2 * np.pi)
        assert a.bin(2)[1] == approx(1 + 3 * np.pi)

        with pytest.raises(IndexError):
            a.bin(-1)  # no underflow
        with pytest.raises(IndexError):
            a.bin(3)

    def test_iter(self):
        a = bh.axis.Regular(2, 1, 2, circular=True)
        ref = [(1, 1.5), (1.5, 2)]
        assert_allclose(a, ref)

    def test_index(self):
        a = bh.axis.Regular(4, 1, 1 + np.pi * 2, circular=True)
        d = 0.5 * np.pi
        assert a.index(0.99 - 4 * d) == 3
        assert a.index(0.99 - 3 * d) == 0
        assert a.index(0.99 - 2 * d) == 1
        assert a.index(0.99 - d) == 2
        assert a.index(0.99) == 3
        assert a.index(1.0) == 0
        assert a.index(1.01) == 0
        assert a.index(0.99 + d) == 0
        assert a.index(1.0 + d) == 1
        assert a.index(1.0 + 2 * d) == 2
        assert a.index(1.0 + 3 * d) == 3
        assert a.index(1.0 + 4 * d) == 0
        assert a.index(1.0 + 5 * d) == 1

    def test_edges_centers_widths(self):
        a = bh.axis.Regular(2, 0, 1, circular=True)
        assert_allclose(a.edges, [0, 0.5, 1])
        assert_allclose(a.centers, [0.25, 0.75])
        assert_allclose(a.widths, [0.5, 0.5])


class TestVariable(Axis):
    def test_init(self):
        # should not raise
        bh.axis.Variable([0, 1])
        bh.axis.Variable((0, 1, 2, 3, 4))
        bh.axis.Variable([0, 1], metadata="va")
        with pytest.raises(TypeError):
            bh.axis.Variable()
        with pytest.raises(ValueError):
            bh.axis.Variable([1])
        with pytest.raises(TypeError):
            bh.axis.Variable(1)
        with pytest.raises(ValueError):
            bh.axis.Variable([1, -1])
        with pytest.raises(ValueError):
            bh.axis.Variable([1, 1])
        with pytest.raises(TypeError):
            bh.axis.Variable(["1", 2])
        with pytest.raises(TypeError):
            bh.axis.Variable([0.0, 1.0, 2.0], bad_keyword="ra")

    def test_traits(self):
        STD_TRAITS = dict(continuous=True, ordered=True)

        ax = bh.axis.Variable([1, 2, 3])
        assert isinstance(ax, bh.axis.Variable)
        assert ax.traits == bh.axis.Traits(underflow=True, overflow=True, **STD_TRAITS)

        ax = bh.axis.Variable([1, 2, 3], overflow=False)
        assert isinstance(ax, bh.axis.Variable)
        assert ax.traits == bh.axis.Traits(underflow=True, **STD_TRAITS)

        ax = bh.axis.Variable([1, 2, 3], underflow=False)
        assert isinstance(ax, bh.axis.Variable)
        assert ax.traits == bh.axis.Traits(overflow=True, **STD_TRAITS)

        ax = bh.axis.Variable([1, 2, 3], underflow=False, overflow=False)
        assert isinstance(ax, bh.axis.Variable)
        assert ax.traits == bh.axis.Traits(**STD_TRAITS)

    def test_equal(self):
        a = bh.axis.Variable([-0.1, 0.2, 0.3])
        assert a == bh.axis.Variable((-0.1, 0.2, 0.3))
        assert a != bh.axis.Variable([0, 0.2, 0.3])
        assert a != bh.axis.Variable([-0.1, 0.1, 0.3])
        assert a != bh.axis.Variable([-0.1, 0.1])

    def test_len(self):
        a = bh.axis.Variable([-0.1, 0.2, 0.3])
        assert len(a) == 2
        assert a.size == 2
        assert a.extent == 4

    def test_repr(self):
        a = bh.axis.Variable([-0.1, 0.2])
        assert repr(a) == "Variable([-0.1, 0.2])"

        a = bh.axis.Variable([-0.1, 0.2], metadata="hi")
        assert repr(a) == "Variable([-0.1, 0.2])"

    def test_getitem(self):
        ref = [-0.1, 0.2, 0.3]
        a = bh.axis.Variable(ref)

        for i in range(2):
            assert_allclose(a.bin(i), ref[i : i + 2])
            assert_allclose(a[i], ref[i : i + 2])

        assert a[-1] == a[1]
        with pytest.raises(IndexError):
            a[2]

        assert_allclose(a[bh.underflow], a.bin(-1))
        assert_allclose(a[bh.overflow], a.bin(2))

        assert a.bin(-1)[0] == -np.inf
        assert a.bin(-1)[1] == ref[0]

        assert a.bin(2)[0] == ref[2]
        assert a.bin(2)[1] == np.inf

        with pytest.raises(IndexError):
            a.bin(-2)
        with pytest.raises(IndexError):
            a.bin(3)

    def test_iter(self):
        ref = [-0.1, 0.2, 0.3]
        a = bh.axis.Variable(ref)
        for i, bin in enumerate(a):
            assert_array_equal(bin, ref[i : i + 2])

    def test_index(self):
        a = bh.axis.Variable([-0.1, 0.2, 0.3])
        assert a.index(-10.0) == -1
        assert a.index(-0.11) == -1
        assert a.index(-0.1) == 0
        assert a.index(0.0) == 0
        assert a.index(0.19) == 0
        assert a.index(0.2) == 1
        assert a.index(0.21) == 1
        assert a.index(0.29) == 1
        assert a.index(0.3) == 2
        assert a.index(0.31) == 2
        assert a.index(10) == 2

    def test_edges_centers_widths(self):
        a = bh.axis.Variable([0, 1, 3])
        assert_allclose(a.edges, [0, 1, 3])
        assert_allclose(a.centers, [0.5, 2])
        assert_allclose(a.widths, [1, 2])


class TestInteger:
    def test_init(self):
        bh.axis.Integer(-1, 2)
        bh.axis.Integer(-1, 2, metadata="foo")
        bh.axis.Integer(-1, 2, underflow=False)
        bh.axis.Integer(-1, 2, underflow=False, overflow=False)
        bh.axis.Integer(-1, 2, growth=True)

        with pytest.raises(TypeError):
            bh.axis.Integer()
        with pytest.raises(TypeError):
            bh.axis.Integer(1)
        with pytest.raises(TypeError):
            bh.axis.Integer("1", 2)
        with pytest.raises(ValueError):
            bh.axis.Integer(2, -1)
        with pytest.raises(TypeError):
            bh.axis.Integer(-1, 2, "foo")
        with pytest.raises(TypeError):
            bh.axis.Integer(20, 30, 40)

    def test_traits(self):
        STD_TRAITS = dict(ordered=True)

        ax = bh.axis.Integer(1, 3)
        assert isinstance(ax, bh.axis.Integer)
        assert ax.traits == bh.axis.Traits(underflow=True, overflow=True, **STD_TRAITS)

        # See https://github.com/boostorg/histogram/issues/305
        ax = bh.axis.Integer(1, 3, overflow=False)
        assert isinstance(ax, bh.axis.Integer)
        assert ax.traits == bh.axis.Traits(underflow=True, **STD_TRAITS)

        # See https://github.com/boostorg/histogram/issues/305
        ax = bh.axis.Integer(1, 3, underflow=False)
        assert isinstance(ax, bh.axis.Integer)
        assert ax.traits == bh.axis.Traits(overflow=True, **STD_TRAITS)

        ax = bh.axis.Integer(1, 3, underflow=False, overflow=False)
        assert isinstance(ax, bh.axis.Integer)
        assert ax.traits == bh.axis.Traits(**STD_TRAITS)

        ax = bh.axis.Integer(1, 3, growth=True)
        assert isinstance(ax, bh.axis.Integer)
        assert ax.traits == bh.axis.Traits(growth=True, **STD_TRAITS)

    def test_equal(self):
        assert bh.axis.Integer(-1, 2) == bh.axis.Integer(-1, 2)
        assert bh.axis.Integer(-1, 2) != bh.axis.Integer(-1, 2, metadata="Other")
        assert bh.axis.Integer(-1, 2, underflow=True) != bh.axis.Integer(
            -1, 2, underflow=False
        )

    def test_len(self, underflow, overflow):
        a = bh.axis.Integer(-1, 3, underflow=underflow, overflow=overflow)
        assert len(a) == 4
        assert a.size == 4
        assert a.extent == 4 + underflow + overflow

    def test_repr(self):
        a = bh.axis.Integer(-1, 1)
        assert repr(a) == "Integer(-1, 1)"

        a = bh.axis.Integer(-1, 1, metadata="hi")
        assert repr(a) == "Integer(-1, 1)"

        a = bh.axis.Integer(-1, 1, underflow=False)
        assert repr(a) == "Integer(-1, 1, underflow=False)"

        a = bh.axis.Integer(-1, 1, overflow=False)
        assert repr(a) == "Integer(-1, 1, overflow=False)"

        a = bh.axis.Integer(-1, 1, growth=True)
        assert repr(a) == "Integer(-1, 1, growth=True)"

    def test_label(self):
        a = bh.axis.Integer(-1, 2, metadata="foo")
        assert a.metadata == "foo"
        a.metadata = "bar"
        assert a.metadata == "bar"

    def test_getitem(self):
        a = bh.axis.Integer(-1, 3)
        ref = [-1, 0, 1, 2]
        for i, r in enumerate(ref):
            assert a.bin(i) == r
            assert a[i] == r
        assert a.bin(-1) == -2
        assert a.bin(4) == 3

        assert_allclose(a[bh.underflow], a.bin(-1))
        assert_allclose(a[bh.overflow], a.bin(4))

    def test_iter(self):
        a = bh.axis.Integer(-1, 3)
        ref = (-1, 0, 1, 2)
        assert_array_equal(a, ref)

    def test_index(self):
        a = bh.axis.Integer(-1, 3)
        assert a.index(-3) == -1
        assert a.index(-2) == -1
        assert a.index(-1) == 0
        assert a.index(0) == 1
        assert a.index(1) == 2
        assert a.index(2) == 3
        assert a.index(3) == 4
        assert a.index(4) == 4

    def test_edges_centers_widths(self):
        a = bh.axis.Integer(1, 3)
        assert_allclose(a.edges, [1, 2, 3])
        assert_allclose(a.centers, [1.5, 2.5])
        assert_allclose(a.widths, [1, 1])


class TestCategory(Axis):
    def test_init(self):
        # should not raise
        bh.axis.IntCategory([1, 2])
        bh.axis.IntCategory({1, 2})
        bh.axis.IntCategory((1, 2), metadata="foo")
        bh.axis.StrCategory(["A", "B"])
        bh.axis.StrCategory({"A", "B"})
        bh.axis.StrCategory("AB")
        bh.axis.StrCategory("AB", metadata="foo")

        with pytest.raises(TypeError):
            bh.axis.IntCategory(["A", "B"])
        with pytest.raises(TypeError):
            bh.axis.StrCategory([1, 2])

        with pytest.raises(TypeError):
            bh.axis.IntCategory([1, 2], "foo")
        with pytest.raises(TypeError):
            bh.axis.StrCategory("AB", "foo")

        with pytest.raises(TypeError):
            bh.axis.StrCategory()
        with pytest.raises(TypeError):
            bh.axis.IntCategory()
        with pytest.raises(TypeError):
            bh.axis.StrCategory([1, "2"])
        with pytest.raises(TypeError):
            bh.axis.IntCategory([1, "2"])
        with pytest.raises(TypeError):
            bh.axis.IntCategory([1, 2, 3], underflow=True)

    def test_traits(self):
        ax = bh.axis.IntCategory([1, 2, 3])
        assert isinstance(ax, bh.axis.IntCategory)
        assert ax.traits == bh.axis.Traits(overflow=True)

        with pytest.warns(FutureWarning):
            assert not ax.options.underflow
            assert ax.options.overflow

        ax = bh.axis.IntCategory([1, 2, 3], growth=True)
        assert isinstance(ax, bh.axis.IntCategory)
        assert ax.traits == bh.axis.Traits(growth=True)

        ax = bh.axis.StrCategory(["1", "2", "3"])
        assert isinstance(ax, bh.axis.StrCategory)
        assert ax.traits == bh.axis.Traits(overflow=True)

        ax = bh.axis.StrCategory(["1", "2", "3"], growth=True)
        assert isinstance(ax, bh.axis.StrCategory)
        assert ax.traits == bh.axis.Traits(growth=True)

    def test_equal(self):
        assert bh.axis.IntCategory([1, 2, 3]) == bh.axis.IntCategory([1, 2, 3])
        assert bh.axis.IntCategory([1, 2, 3]) != bh.axis.IntCategory([1, 3, 2])
        assert bh.axis.StrCategory(["A", "B"]) == bh.axis.StrCategory("AB")
        assert bh.axis.StrCategory(["A", "B"]) != bh.axis.StrCategory("BA")

    @pytest.mark.parametrize("ref", ([1, 2, 3], "ABC"))
    def test_len(self, ref, growth):
        Cat = bh.axis.StrCategory if isinstance(ref[0], str) else bh.axis.IntCategory
        a = Cat(ref, growth=growth)
        assert len(a) == 3
        assert a.size == 3
        assert a.extent == 3 if growth else 4

    def test_repr(self):
        ax = bh.axis.IntCategory([1, 2, 3])
        assert repr(ax) == "IntCategory([1, 2, 3])"

        ax = bh.axis.IntCategory([1, 2, 3], metadata="foo")
        assert repr(ax) == "IntCategory([1, 2, 3])"

        ax = bh.axis.StrCategory("ABC", metadata="foo")
        # If unicode is the default (Python 3, generally)
        if type("") == type(u""):  # noqa: E721
            assert repr(ax) == "StrCategory(['A', 'B', 'C'])"
        else:
            assert repr(ax) == "StrCategory([u'A', u'B', u'C'])"

    @pytest.mark.parametrize("ref", ([1, 2, 3], "ABC"))
    def test_getitem(self, ref, growth):
        Cat = bh.axis.StrCategory if isinstance(ref[0], str) else bh.axis.IntCategory

        a = Cat(ref, growth=growth)

        for i in range(3):
            assert a.bin(i) == ref[i]
            assert a[i] == ref[i]

        assert a[-1] == a[2]
        with pytest.raises(IndexError):
            a[3]

        with pytest.raises(IndexError):
            a.bin(-1)

        if growth:
            with pytest.raises(IndexError):
                a.bin(3)
        else:
            assert a.bin(3) is None

    @pytest.mark.parametrize("ref", ([1, 2, 3], ("A", "B", "C")))
    def test_iter(self, ref, growth):
        Cat = bh.axis.StrCategory if isinstance(ref[0], str) else bh.axis.IntCategory
        a = Cat(ref, growth=growth)
        assert_array_equal(a, ref)

    @pytest.mark.parametrize(
        "ref", ([1, 2, 3, 4], ("A", "B", "C", "D")), ids=("int", "str")
    )
    def test_index(self, ref, growth):
        Cat = bh.axis.StrCategory if isinstance(ref[0], str) else bh.axis.IntCategory
        a = Cat(ref[:-1], growth=growth)
        for i, r in enumerate(ref):
            assert a.index(r) == i
        assert_array_equal(a.index(ref), [0, 1, 2, 3])
        assert_array_equal(a.index(np.reshape(ref, (2, 2))), [[0, 1], [2, 3]])

    @pytest.mark.parametrize("ref", ([1, 2, 3], ("A", "B", "C")))
    def test_value(self, ref, growth):
        Cat = bh.axis.StrCategory if isinstance(ref[0], str) else bh.axis.IntCategory
        a = Cat(ref, growth=growth)
        for i, r in enumerate(ref):
            assert a.value(i) == r
        assert_array_equal(a.value(range(3)), ref)
        assert a.value(3) is None
        assert_array_equal(a.value((0, 3)), [ref[0], None])
        assert_array_equal(
            a.value(np.array((0, 1, 2, 3))), [ref[0], ref[1], ref[2], None]
        )
        # may be added in the future
        with pytest.raises(ValueError):
            a.value([[2], [2]])

    @pytest.mark.parametrize("ref", ([1, 2, 3], "ABC"))
    def test_edges_centers_widths(self, ref, growth):
        Cat = bh.axis.StrCategory if isinstance(ref[0], str) else bh.axis.IntCategory
        a = Cat(ref, growth=growth)
        assert_allclose(a.edges, [0, 1, 2, 3])
        assert_allclose(a.centers, [0.5, 1.5, 2.5])
        assert_allclose(a.widths, [1, 1, 1])


class TestBoolean:
    def test_init(self):
        bh.axis.Boolean()
        bh.axis.Boolean(metadata="foo")

        with pytest.raises(TypeError):
            bh.axis.Boolean(1)

    def test_traits(self):
        ax = bh.axis.Boolean()
        assert isinstance(ax, bh.axis.Boolean)

        assert ax.traits == bh.axis.Traits(ordered=True)

    def test_equal(self):
        assert bh.axis.Boolean() == bh.axis.Boolean()
        assert bh.axis.Boolean(metadata="hi") == bh.axis.Boolean(metadata="hi")
        assert bh.axis.Boolean(metadata="hi") != bh.axis.Boolean()
        assert bh.axis.Boolean(metadata="hi") != bh.axis.Boolean(metadata="ho")

    def test_len(self):
        a = bh.axis.Boolean()
        assert len(a) == 2
        assert a.size == 2
        assert a.extent == 2

    def test_repr(self):
        a = bh.axis.Boolean()
        assert repr(a) == "Boolean()"

        a = bh.axis.Boolean(metadata="hi")
        assert repr(a) == "Boolean()"

    def test_label(self):
        a = bh.axis.Boolean(metadata="foo")
        assert a.metadata == "foo"
        a.metadata = "bar"
        assert a.metadata == "bar"

    def test_getitem(self):
        a = bh.axis.Boolean()
        ref = [False, True]
        for i, r in enumerate(ref):
            assert a.bin(i) == r
            assert a[i] == r
        assert a.bin(0) == 0
        assert a.bin(1) == 1

    def test_iter(self):
        a = bh.axis.Boolean()
        ref = (False, True)
        assert_array_equal(a, ref)

    def test_index(self):
        a = bh.axis.Boolean()
        assert a.index(False) == 0
        assert a.index(True) == 1

    def test_edges_centers_widths(self):
        a = bh.axis.Boolean()
        assert_allclose(a.edges, [0.0, 1.0, 2.0])
        assert_allclose(a.centers, [0.5, 1.5])
        assert_allclose(a.widths, [1, 1])
