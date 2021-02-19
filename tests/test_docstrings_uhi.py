# -*- coding: utf-8 -*-
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Cannot import protocol on Python 2"
)


import plottable  # noqa: E402

import boost_histogram as bh  # noqa: E402


def test_plotting_protocol():
    assert (
        plottable.PlottableHistogram.values.__doc__.strip()
        in bh.Histogram.values.__doc__
    )
    assert (
        plottable.PlottableHistogram.variances.__doc__.strip()[:300]
        in bh.Histogram.variances.__doc__
    )
    assert (
        plottable.PlottableHistogram.counts.__doc__.strip()[:1000]
        in bh.Histogram.counts.__doc__
    )
