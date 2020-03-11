# You can enter maximum compatibility mode by typing:
# import boost_histogram.cpp as bh
# However, this is a test so we will import both.
import boost_histogram as bh
import boost_histogram.cpp as bhc
from boost_histogram.cpp.algorithm import slice_mode
from numpy.testing import assert_array_equal
import numpy as np
import pytest


def test_usage_bh():
    h = bhc.histogram(bhc.axis.regular(10, 0, 1), bhc.axis.str_category(["one", "two"]))
    assert h.axis(0) == bhc.axis.regular(10, 0, 1)
    assert h.axis(1) == bhc.axis.str_category(["one", "two"])

    h(0, "one")

    assert h.at(0, 0) == 1.0


def test_convert_bh():
    h = bhc.histogram(
        bh.Histogram(bh.axis.Regular(10, 0, 1), bh.axis.StrCategory(["one", "two"]))
    )
    assert hasattr(h, "axis")
    assert not hasattr(h, "axes")

    h = bh.Histogram(h)

    # Current warning workaround. Enable when removed:
    # assert not hasattr(h, "axis")
    assert hasattr(h, "axes")


def test_str():
    # Mixing bh/bhc is fine; histogram correctly always returns matching axis,
    # axis returns matching transform, etc.
    h = bhc.histogram(bh.axis.Regular(4, 0, 4))
    assert repr(str(h)) == repr(
        """              +--------------------------------------------------------------+
[-inf,   0) 0 |                                                              |
[   0,   1) 0 |                                                              |
[   1,   2) 0 |                                                              |
[   2,   3) 0 |                                                              |
[   3,   4) 0 |                                                              |
[   4, inf) 0 |                                                              |
              +--------------------------------------------------------------+"""
    )


def test_repr():
    h = bhc.histogram(bh.axis.Regular(4, 0, 4))
    assert (
        repr(h)
        == """histogram(
  regular(4, 0, 4, metadata="None", options=underflow | overflow),
  storage=double())"""
    )


def test_axis_reprs():
    ax = bhc.axis.regular(4, 0, 4)
    assert repr(ax) == 'regular(4, 0, 4, metadata="None", options=underflow | overflow)'
    assert repr(type(ax)) == "<class 'boost_histogram.cpp.axis.regular'>"


def test_storage_repr():
    h = bhc.histogram(bhc.axis.regular(10, 0, 1))
    assert repr(h._storage_type()) == "double()"
    assert repr(h._storage_type) == "<class 'boost_histogram.cpp.storage.double'>"


def test_transform_repr():
    ax = bhc.axis.regular(8, 0, 3, transform=bhc.axis.transform.pow(2))
    assert repr(ax.transform()), "pow(2)"
    assert repr(
        type(ax.transform())
    ), "<class 'boost_histogram.cpp.axis.transform.pow'>"


def test_shrink():
    from boost_histogram.cpp.algorithm import reduce, shrink

    h = bhc.histogram(bhc.axis.regular(4, 1, 5))
    np.asarray(h)[:] = 1

    hs = reduce(h, shrink(0, 2, 3))
    assert hs.axis(0) == bhc.axis.regular(1, 2, 3)
    assert_array_equal(hs, [1])
    assert hs.at(-1) == 1
    assert hs.at(1) == 2

    hs2 = reduce(h, shrink(2, 3))
    assert hs == hs2


def test_crop():
    from boost_histogram.cpp.algorithm import reduce, crop

    h = bhc.histogram(bhc.axis.regular(4, 1, 5))
    np.asarray(h)[:] = 1

    hs = reduce(h, crop(0, 2, 3))
    assert hs.axis(0) == bhc.axis.regular(1, 2, 3)
    assert_array_equal(hs, [1])
    assert hs.at(-1) == 0
    assert hs.at(1) == 0

    hs2 = reduce(h, crop(2, 3))
    assert hs == hs2


@pytest.mark.parametrize("mode", (slice_mode.shrink, slice_mode.crop))
def test_slice(mode):
    from boost_histogram.cpp.algorithm import reduce, slice

    h = bhc.histogram(bhc.axis.regular(4, 1, 5))
    np.asarray(h)[:] = 1
    assert_array_equal(h, [1, 1, 1, 1])

    hs = reduce(h, slice(0, 1, 2, mode=mode))
    assert hs.axis(0) == bhc.axis.regular(1, 2, 3)
    assert_array_equal(hs, [1])
    assert hs.at(-1) == (1 if mode == slice_mode.shrink else 0)
    assert hs.at(1) == (2 if mode == slice_mode.shrink else 0)

    hs2 = reduce(h, slice(1, 2, mode=mode))
    assert hs == hs2


def test_rebin():
    from boost_histogram.cpp.algorithm import reduce, rebin

    h = bhc.histogram(bhc.axis.regular(4, 1, 5))
    np.asarray(h)[:] = 1
    assert_array_equal(h, [1, 1, 1, 1])

    hs = reduce(h, rebin(0, 4))
    assert hs.axis(0) == bhc.axis.regular(1, 1, 5)
    assert_array_equal(hs, [4])

    hs2 = reduce(h, rebin(4))
    assert hs == hs2


def test_shrink_and_rebin():
    from boost_histogram.cpp.algorithm import reduce, shrink_and_rebin

    h = bhc.histogram(bhc.axis.regular(5, 0, 5))
    np.asarray(h)[:] = 1
    hs = reduce(h, shrink_and_rebin(0, 1, 3, 2))
    assert hs.axis(0) == bhc.axis.regular(1, 1, 3)
    assert_array_equal(hs, [2])
    hs2 = reduce(h, shrink_and_rebin(1, 3, 2))
    assert hs == hs2


def test_crop_and_rebin():
    from boost_histogram.cpp.algorithm import reduce, crop_and_rebin

    h = bhc.histogram(bhc.axis.regular(5, 0, 5))
    np.asarray(h)[:] = 1
    hs = reduce(h, crop_and_rebin(0, 1, 3, 2))
    assert hs.axis(0) == bhc.axis.regular(1, 1, 3)
    assert_array_equal(hs, [2])
    hs2 = reduce(h, crop_and_rebin(1, 3, 2))
    assert hs == hs2


def test_slice_and_rebin():
    from boost_histogram.cpp.algorithm import reduce, slice_and_rebin

    h = bhc.histogram(bhc.axis.regular(5, 0, 5))
    np.asarray(h)[:] = 1
    hs = reduce(h, slice_and_rebin(0, 1, 3, 2))
    assert hs.axis(0) == bhc.axis.regular(1, 1, 3)
    assert_array_equal(hs, [2])
    hs2 = reduce(h, slice_and_rebin(1, 3, 2))
    assert hs == hs2
