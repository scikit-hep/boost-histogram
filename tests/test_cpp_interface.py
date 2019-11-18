# You can enter maximum compatibility mode by typing:
# import boost_histogram.cpp as bh
# However, this is a test so we will import both.
import boost_histogram as bh
import boost_histogram.cpp as bhc
import pytest


def test_usage_bh():
    h = bhc.histogram(bhc.axis.regular(10, 0, 1), bhc.axis.category(["one", "two"]))
    assert h.axis(0) == bhc.axis.regular(10, 0, 1)
    assert h.axis(1) == bhc.axis.category(["one", "two"])

    h(0, "one")

    assert h.at(0, 0) == 1.0


def test_convert_bh():
    h = bhc.histogram(
        bh.Histogram(bh.axis.Regular(10, 0, 1), bh.axis.Category(["one", "two"]))
    )
    assert hasattr(h, "axis")
    assert not hasattr(h, "axes")

    h = bh.Histogram(h)

    # Current warning workaround. Enable when removed:
    # assert not hasattr(h, "axis")
    assert hasattr(h, "axes")


def test_reprs():
    # Mixing bh/bhc is fine; histogram correctly always returns matching axis,
    # axis returns matching transform, etc.
    h = bhc.histogram(bh.axis.Regular(4, 0, 4))
    assert (
        repr(h)
        == """\
histogram(regular(4, 0, 4, metadata="None", options=underflow | overflow))
              +--------------------------------------------------------------+
[-inf,   0) 0 |                                                              |
[   0,   1) 0 |                                                              |
[   1,   2) 0 |                                                              |
[   2,   3) 0 |                                                              |
[   3,   4) 0 |                                                              |
[   4, inf) 0 |                                                              |
              +--------------------------------------------------------------+
"""
    )

    assert (
        repr(h.axis(0))
        == 'regular(4, 0, 4, metadata="None", options=underflow | overflow)'
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
