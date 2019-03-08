import histogram as bh
import numpy as np


def test_1D_fill_unlimited():
    bins = 10
    ranges = (0, 1)

    vals = (.15, .25, .25)

    hist = bh.hist.regular_unlimited([
        bh.axis.regular(bins, *ranges)
        ])
    hist.fill(vals)


def test_1D_fill_int():
    bins = 10
    ranges = (0, 1)

    vals = (.15, .25, .25)

    hist = bh.hist.regular_int([
        bh.axis.regular(bins, *ranges)
        ])
    hist.fill(vals)

    assert [hist.at(i) for i in range(10)] == [0, 1, 2, 0, 0, 0, 0, 0, 0, 0]

def test_2D_fill_int():
    bins = (10, 10)
    ranges = ((0, 1), (0, 1))

    vals = ((.15, .25, .25), (.35, .45, .45))

    hist = bh.hist.regular_int([
        bh.axis.regular(bins[0], *ranges[0]),
        bh.axis.regular(bins[1], *ranges[1]),
        ])
    hist.fill(vals)

    H, *ledges = np.histogram2d(*vals, bins=bins, range=ranges)

    assert [hist.at(i // 10, i % 10) for i in range(100)] == [H[i // 10, i % 10] for i in range(100)]
