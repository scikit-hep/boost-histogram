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

    assert not hasattr(h, "axis")
    assert hasattr(h, "axes")
