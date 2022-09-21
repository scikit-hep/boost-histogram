from __future__ import annotations

import boost_histogram as bh


def test_subclass():
    NEW_FAMILY = object()

    class MyHist(bh.Histogram, family=NEW_FAMILY):
        pass

    class MyRegular(bh.axis.Regular, family=NEW_FAMILY):
        __slots__ = ()

    class MyIntStorage(bh.storage.Int64, family=NEW_FAMILY):
        pass

    class MyPowTransform(bh.axis.transform.Pow, family=NEW_FAMILY):
        pass

    h = MyHist(MyRegular(10, 0, 2, transform=MyPowTransform(2)), storage=MyIntStorage())

    assert type(h) == MyHist
    assert h.storage_type == MyIntStorage
    assert type(h.axes[0]) == MyRegular
    assert type(h.axes[0].transform) == MyPowTransform


def test_subclass_hist_only():
    class MyHist(bh.Histogram):
        pass

    h = MyHist(bh.axis.Regular(10, 0, 2))

    assert type(h) == MyHist
    assert type(h.axes[0]) == bh.axis.Regular


def test_copy():
    class MyHist(bh.Histogram):
        def __init__(self, var, bins, weight, **kwargs):
            super().__init__(
                bh.axis.Regular(*bins), storage=bh.storage.Weight(), **kwargs
            )

            self.fill(var, weight=weight)

    b = (2, 0, 1)
    v = [0.1, 0.5, 0.9]
    w = [1, 0.5, 1]
    hist = MyHist(v, b, w)

    hist.copy()
