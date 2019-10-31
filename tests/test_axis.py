import pytest
from pytest import approx

from boost_histogram.axis import options
from boost_histogram.axis import (
    regular,
    regular_log,
    regular_sqrt,
    regular_pow,
    circular,
    variable,
    integer,
    category,
)

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

import copy
import abc

# compatible with Python 2 *and* 3:
ABC = abc.ABCMeta("ABC", (object,), {"__slots__": ()})


@pytest.mark.parametrize(
    "axis_and_args",
    [
        (regular, (1, 2, 3, "")),
        (regular, (1, 2, 3, "u")),
        (regular, (1, 2, 3, "o")),
        (regular, (1, 2, 3, "uo")),
        (regular, (1, 2, 3, "uog")),
        (circular, (1, 2, 3, "")),
        (regular_log, (1, 2, 3, "")),
        (regular_sqrt, (1, 2, 3, "")),
        (regular_pow, (1, 2, 3, 1, "")),
        (variable, ((1, 2, 3), "")),
        (variable, ((1, 2, 3), "u")),
        (variable, ((1, 2, 3), "o")),
        (variable, ((1, 2, 3), "uo")),
        (variable, ((1, 2, 3), "uog")),
        (integer, (1, 2, "")),
        (integer, (1, 2, "u")),
        (integer, (1, 2, "o")),
        (integer, (1, 2, "uo")),
        (integer, (1, 2, "g")),
        (category, ((1, 2, 3), "")),
        (category, ((1, 2, 3), "g")),
        (category, ("ABC", "")),
        (category, ("ABC", "g")),
    ],
)
def test_metadata(axis_and_args):
    axis, args_opt = axis_and_args
    args = args_opt[:-1]
    opt = args_opt[-1]
    for m in ("foo", 64, {"one": 1}):
        kwargs = {
            "underflow": "u" in opt,
            "overflow": "o" in opt,
            "growth": "g" in opt,
            "metadata": m,
        }
        if axis is category:
            del kwargs["underflow"]
            del kwargs["overflow"]
        if axis in (circular, regular_log, regular_sqrt, regular_pow):
            del kwargs["underflow"]
            del kwargs["overflow"]
            del kwargs["growth"]
        assert axis(*args, **kwargs).metadata is m
        mcopy = copy.deepcopy(m)
        # assert axis(*args, m).metadata is not mcopy
        assert axis(*args, **kwargs).metadata == m
        assert axis(*args, **kwargs).metadata == mcopy
        assert axis(*args, **kwargs).metadata != "bar"
        assert axis(*args, **kwargs) == axis(*args, **kwargs)
        assert axis(*args, **kwargs) != axis(*args, metadata="bar")


class Axis(ABC):
    @abc.abstractmethod
    def test_init(self):
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
        regular(1, 1.0, 2.0)
        regular(1, 1.0, 2.0, metadata="ra")
        regular(1, 1.0, 2.0, underflow=False)
        regular(1, 1.0, 2.0, underflow=False, overflow=False, metadata="ra")
        regular(1, 1.0, 2.0, metadata=0)
        regular_log(1, 1.0, 2.0)
        regular_sqrt(1, 1.0, 2.0)
        regular_pow(1, 1.0, 2.0, 1.5)

        with pytest.raises(TypeError):
            regular()
        with pytest.raises(TypeError):
            regular(overflow=False, underflow=False)
        with pytest.raises(TypeError):
            regular(1)
        with pytest.raises(TypeError):
            regular(1, 1.0)
        with pytest.raises(ValueError):
            regular(0, 1.0, 2.0)
        with pytest.raises(TypeError):
            regular("1", 1.0, 2.0)
        with pytest.raises(Exception):
            regular(-1, 1.0, 2.0)

        with pytest.raises(ValueError):
            regular(1, 1.0, 1.0)

        with pytest.raises(TypeError):
            regular(1, 1.0, 2.0, bad_keyword="ra")
        with pytest.raises(TypeError):
            regular_pow(1, 1.0, 2.0)

        ax = regular(1, 2, 3)
        assert isinstance(ax, regular)
        assert ax.options == options(underflow=True, overflow=True)

        ax = regular(1, 2, 3, overflow=False)
        assert isinstance(ax, regular)
        assert ax.options == options(underflow=True)

        ax = regular(1, 2, 3, underflow=False)
        assert isinstance(ax, regular)
        assert ax.options == options(overflow=True)

        ax = regular(1, 2, 3, underflow=False, overflow=False)
        assert isinstance(ax, regular)
        assert ax.options == options()

        ax = regular(1, 2, 3, growth=True)
        assert isinstance(ax, regular)
        assert ax.options == options(underflow=True, overflow=True, growth=True)

    def test_equal(self):
        a = regular(4, 1.0, 2.0)
        assert a == regular(4, 1.0, 2.0)
        assert a != regular(3, 1.0, 2.0)
        assert a != regular(4, 1.1, 2.0)
        assert a != regular(4, 1.0, 2.1)

        # metadata compare
        assert regular(1, 2, 3, metadata=1) == regular(1, 2, 3, metadata=1)
        assert regular(1, 2, 3, metadata=1) != regular(1, 2, 3, metadata="1")
        assert regular(1, 2, 3, metadata=1) != regular(1, 2, 3, metadata=[1])

    def test_len(self):
        a = regular(4, 1.0, 2.0)
        assert len(a) == 4
        assert a.size == 4
        assert a.extent == 6

    def test_repr(self):
        ax = regular(4, 1.1, 2.2)
        assert repr(ax) == "regular(4, 1.1, 2.2)"

        ax = regular(4, 1.1, 2.2, metadata="ra")
        assert repr(ax) == 'regular(4, 1.1, 2.2, metadata="ra")'

        ax = regular(4, 1.1, 2.2, underflow=False)
        assert repr(ax) == "regular(4, 1.1, 2.2, underflow=False)"

        ax = regular(4, 1.1, 2.2, metadata="ra", overflow=False)
        assert repr(ax) == 'regular(4, 1.1, 2.2, metadata="ra", overflow=False)'

        ax = regular_log(4, 1.1, 2.2)
        assert repr(ax) == "regular_log(4, 1.1, 2.2)"

        ax = regular_sqrt(3, 1.1, 2.2)
        assert repr(ax) == "regular_sqrt(3, 1.1, 2.2)"

        ax = regular_pow(4, 1.1, 2.2, 0.5)
        assert repr(ax) == "regular_pow(4, 1.1, 2.2, power=0.5)"

    def test_getitem(self):
        a = regular(2, 1.0, 2.0)
        ref = [1.0, 1.5, 2.0]
        for i in range(2):
            assert_allclose(a.bin(i), ref[i : i + 2])
            assert_allclose(a[i], ref[i : i + 2])

        assert a[-1] == a[1]
        with pytest.raises(IndexError):
            a[2]

        assert a.bin(-1)[0] == -np.inf
        assert a.bin(2)[1] == np.inf

        with pytest.raises(IndexError):
            a.bin(-2)
        with pytest.raises(IndexError):
            a.bin(3)

    def test_iter(self):
        a = regular(2, 1.0, 2.0)
        ref = [(1.0, 1.5), (1.5, 2.0)]
        assert_allclose(a, ref)

    def test_index(self):
        a = regular(4, 1.0, 2.0)

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
        a = regular(4, 2.0, 1.0)

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

    def test_log_transform(self):
        a = regular_log(2, 1e0, 1e2)

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
        a = regular_pow(2, 1.0, 9.0, power=0.5)

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
        a = regular(2, 0, 1)
        assert_allclose(a.edges, [0, 0.5, 1])
        assert_allclose(a.centers, [0.25, 0.75])
        assert_allclose(a.widths, [0.5, 0.5])


class TestCircular(Axis):
    def test_init(self):
        # Should not throw
        circular(1, 2, 3)
        circular(1, 2, 3, metadata="pa")

        with pytest.raises(TypeError):
            circular(1, 2, 3, "pa")

        with pytest.raises(TypeError):
            circular()
        with pytest.raises(TypeError):
            circular(1)
        with pytest.raises(TypeError):
            circular(1, -1)
        with pytest.raises(Exception):
            circular(-1)
        with pytest.raises(TypeError):
            circular(1, 1.0, metadata=1)
        with pytest.raises(TypeError):
            circular("1")

    def test_equal(self):
        a = circular(4, 0.0, 1.0)
        assert a == circular(4, 0, 1)
        assert a != circular(2, 0, 1)
        assert isinstance(a, regular)

    def test_len(self):
        assert len(circular(4, 0.0, 1.0)) == 4
        assert circular(4, 0.0, 1.0).size == 4
        assert circular(4, 0.0, 1.0).extent == 5

    def test_repr(self):
        ax = circular(4, 1.1, 2.2)
        assert repr(ax) == "circular(4, 1.1, 2.2)"

        ax = circular(4, 1.1, 2.2, metadata="hi")
        assert repr(ax) == 'circular(4, 1.1, 2.2, metadata="hi")'

    def test_getitem(self):
        a = circular(2, 1, 1 + np.pi * 2)
        ref = [1.0, 1.0 + np.pi, 1.0 + 2.0 * np.pi]
        for i in range(2):
            assert_allclose(a.bin(i), ref[i : i + 2])
            assert_allclose(a[i], ref[i : i + 2])

        assert a[-1] == a[1]
        with pytest.raises(IndexError):
            a[2]

        assert a.bin(2)[0] == approx(1 + 2 * np.pi)
        assert a.bin(2)[1] == approx(1 + 3 * np.pi)

        with pytest.raises(IndexError):
            a.bin(-1)  # no underflow
        with pytest.raises(IndexError):
            a.bin(3)

    def test_iter(self):
        a = circular(2, 1, 2)
        ref = [(1, 1.5), (1.5, 2)]
        assert_allclose(a, ref)

    def test_index(self):
        a = circular(4, 1, 1 + np.pi * 2)
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
        a = circular(2, 0, 1)
        assert_allclose(a.edges, [0, 0.5, 1])
        assert_allclose(a.centers, [0.25, 0.75])
        assert_allclose(a.widths, [0.5, 0.5])


class TestVariable(Axis):
    def test_init(self):
        # should not raise
        variable([0, 1])
        variable((0, 1, 2, 3, 4))
        variable([0, 1], metadata="va")
        with pytest.raises(TypeError):
            variable()
        with pytest.raises(ValueError):
            variable([1])
        with pytest.raises(TypeError):
            variable(1)
        with pytest.raises(ValueError):
            variable([1, -1])
        with pytest.raises(ValueError):
            variable([1, 1])
        with pytest.raises(TypeError):
            variable(["1", 2])
        with pytest.raises(TypeError):
            variable([0.0, 1.0, 2.0], bad_keyword="ra")

        ax = variable([1, 2, 3])
        assert isinstance(ax, variable)
        assert ax.options == options(underflow=True, overflow=True)

        ax = variable([1, 2, 3], overflow=False)
        assert isinstance(ax, variable)
        assert ax.options == options(underflow=True)

        ax = variable([1, 2, 3], underflow=False)
        assert isinstance(ax, variable)
        assert ax.options == options(overflow=True)

        ax = variable([1, 2, 3], underflow=False, overflow=False)
        assert isinstance(ax, variable)
        assert ax.options == options()

    def test_equal(self):
        a = variable([-0.1, 0.2, 0.3])
        assert a == variable((-0.1, 0.2, 0.3))
        assert a != variable([0, 0.2, 0.3])
        assert a != variable([-0.1, 0.1, 0.3])
        assert a != variable([-0.1, 0.1])

    def test_len(self):
        a = variable([-0.1, 0.2, 0.3])
        assert len(a) == 2
        assert a.size == 2
        assert a.extent == 4

    def test_repr(self):
        a = variable([-0.1, 0.2])
        assert repr(a) == "variable([-0.1, 0.2])"

        a = variable([-0.1, 0.2], metadata="hi")
        assert repr(a) == 'variable([-0.1, 0.2], metadata="hi")'

    def test_getitem(self):
        ref = [-0.1, 0.2, 0.3]
        a = variable(ref)

        for i in range(2):
            assert_allclose(a.bin(i), ref[i : i + 2])
            assert_allclose(a[i], ref[i : i + 2])

        assert a[-1] == a[1]
        with pytest.raises(IndexError):
            a[2]

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
        a = variable(ref)
        for i, bin in enumerate(a):
            assert_array_equal(bin, ref[i : i + 2])

    def test_index(self):
        a = variable([-0.1, 0.2, 0.3])
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
        a = variable([0, 1, 3])
        assert_allclose(a.edges, [0, 1, 3])
        assert_allclose(a.centers, [0.5, 2])
        assert_allclose(a.widths, [1, 2])


class TestInteger:
    def test_init(self):
        integer(-1, 2)
        integer(-1, 2, metadata="foo")
        integer(-1, 2, underflow=False)
        integer(-1, 2, underflow=False, overflow=False)
        integer(-1, 2, growth=True)

        with pytest.raises(TypeError):
            integer()
        with pytest.raises(TypeError):
            integer(1)
        with pytest.raises(TypeError):
            integer("1", 2)
        with pytest.raises(ValueError):
            integer(2, -1)
        with pytest.raises(TypeError):
            integer(-1, 2, "foo")
        with pytest.raises(TypeError):
            integer(20, 30, 40)

        ax = integer(1, 3)
        assert isinstance(ax, integer)
        assert ax.options == options(underflow=True, overflow=True)

        ax = integer(1, 3, overflow=False)
        assert isinstance(ax, integer)
        assert ax.options == options(underflow=True)

        ax = integer(1, 3, underflow=False)
        assert isinstance(ax, integer)
        assert ax.options == options(overflow=True)

        ax = integer(1, 3, underflow=False, overflow=False)
        assert isinstance(ax, integer)
        assert ax.options == options()

        ax = integer(1, 3, growth=True)
        assert isinstance(ax, integer)
        assert ax.options == options(growth=True)

    def test_equal(self):
        assert integer(-1, 2) == integer(-1, 2)
        assert integer(-1, 2) != integer(-1, 2, metadata="Other")
        assert integer(-1, 2, underflow=True) != integer(-1, 2, underflow=False)

    @pytest.mark.parametrize("underflow", [0, 1])
    @pytest.mark.parametrize("overflow", [0, 1])
    def test_len(self, underflow, overflow):
        a = integer(-1, 3, underflow=underflow, overflow=overflow)
        assert len(a) == 4
        assert a.size == 4
        assert a.extent == 4 + underflow + overflow

    def test_repr(self):
        a = integer(-1, 1)
        assert repr(a) == "integer(-1, 1)"

        a = integer(-1, 1, metadata="hi")
        assert repr(a) == 'integer(-1, 1, metadata="hi")'

        a = integer(-1, 1, underflow=False)
        assert repr(a) == "integer(-1, 1, underflow=False)"

        a = integer(-1, 1, overflow=False)
        assert repr(a) == "integer(-1, 1, overflow=False)"

        a = integer(-1, 1, growth=True)
        assert repr(a) == "integer(-1, 1, growth=True)"

    def test_label(self):
        a = integer(-1, 2, metadata="foo")
        assert a.metadata == "foo"
        a.metadata = "bar"
        assert a.metadata == "bar"

    def test_getitem(self):
        a = integer(-1, 3)
        ref = [-1, 0, 1, 2]
        for i, r in enumerate(ref):
            assert a.bin(i) == r
            assert a[i] == r
        assert a.bin(-1) == -2
        assert a.bin(4) == 3

    def test_iter(self):
        a = integer(-1, 3)
        ref = (-1, 0, 1, 2)
        assert_array_equal(a, ref)

    def test_index(self):
        a = integer(-1, 3)
        assert a.index(-3) == -1
        assert a.index(-2) == -1
        assert a.index(-1) == 0
        assert a.index(0) == 1
        assert a.index(1) == 2
        assert a.index(2) == 3
        assert a.index(3) == 4
        assert a.index(4) == 4

    def test_edges_centers_widths(self):
        a = integer(1, 3)
        assert_allclose(a.edges, [1, 2, 3])
        assert_allclose(a.centers, [1.5, 2.5])
        assert_allclose(a.widths, [1, 1])


class TestCategory(Axis):
    def test_init(self):
        # should not raise
        category([1, 2])
        category((1, 2), metadata="foo")
        category(["A", "B"])
        category("AB")
        category("AB", metadata="foo")

        with pytest.raises(TypeError):
            category([1, 2], "foo")
        with pytest.raises(TypeError):
            category("AB", "foo")

        with pytest.raises(TypeError):
            category()
        with pytest.raises(TypeError):
            category([1, "2"])
        with pytest.raises(TypeError):
            category([1, 2, 3], underflow=True)

        ax = category([1, 2, 3])
        assert isinstance(ax, category)
        assert ax.options == options(overflow=True)

        ax = category([1, 2, 3], growth=True)
        assert isinstance(ax, category)
        assert ax.options == options(growth=True)

        ax = category(["1", "2", "3"])
        assert isinstance(ax, category)
        assert ax.options == options(overflow=True)

        ax = category(["1", "2", "3"], growth=True)
        assert isinstance(ax, category)
        assert ax.options == options(growth=True)

    def test_equal(self):
        assert category([1, 2, 3]) == category([1, 2, 3])
        assert category([1, 2, 3]) != category([1, 3, 2])
        assert category(["A", "B"]) == category("AB")
        assert category(["A", "B"]) != category("BA")

    @pytest.mark.parametrize("ref", ([1, 2, 3], "ABC"))
    @pytest.mark.parametrize("growth", (False, True))
    def test_len(self, ref, growth):
        a = category(ref, growth=growth)
        assert len(a) == 3
        assert a.size == 3
        assert a.extent == 3 if growth else 4

    def test_repr(self):
        ax = category([1, 2, 3])
        assert repr(ax) == "category([1, 2, 3])"

        ax = category([1, 2, 3], metadata="foo")
        assert repr(ax) == 'category([1, 2, 3], metadata="foo")'

        ax = category("ABC", metadata="foo")
        assert repr(ax) == 'category(["A", "B", "C"], metadata="foo")'

    @pytest.mark.parametrize("ref", ([1, 2, 3], "ABC"))
    @pytest.mark.parametrize("growth", (False, True))
    def test_getitem(self, ref, growth):
        a = category(ref, growth=growth)

        for i in range(3):
            assert a.bin(i) == ref[i]
            assert a[i] == ref[i]

        assert a[-1] == a[2]
        with pytest.raises(IndexError):
            a[3]

        with pytest.raises(IndexError):
            a.bin(-1)
        # Even if category axis has overflow enabled, we cannot return a bin value for the overflow,
        # because it is not clear what that value should be. So we raise an IndexError when this bin is accessed.
        with pytest.raises(IndexError):
            a.bin(3)

    @pytest.mark.parametrize("ref", ([1, 2, 3], ("A", "B", "C")))
    @pytest.mark.parametrize("growth", (False, True))
    def test_iter(self, ref, growth):
        a = category(ref, growth=growth)
        assert_array_equal(a, ref)

    @pytest.mark.parametrize("ref", ([1, 2, 3], ("A", "B", "C")))
    @pytest.mark.parametrize("growth", (False, True))
    def test_index(self, ref, growth):
        a = category(ref, growth=growth)
        for i, r in enumerate(ref):
            assert a.index(r) == i
        assert_array_equal(a.index(ref), [0, 1, 2])

    @pytest.mark.parametrize("ref", ([1, 2, 3], "ABC"))
    @pytest.mark.parametrize("growth", (False, True))
    def test_edges_centers_widths(self, ref, growth):
        a = category(ref, growth=growth)
        assert_allclose(a.edges, [0, 1, 2, 3])
        assert_allclose(a.centers, [0.5, 1.5, 2.5])
        assert_allclose(a.widths, [1, 1, 1])
