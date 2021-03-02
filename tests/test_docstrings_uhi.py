import plottable

import boost_histogram as bh


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
