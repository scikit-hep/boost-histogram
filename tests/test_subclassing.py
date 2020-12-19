# -*- coding: utf-8 -*-
import boost_histogram as bh


def test_subclass():
    NEW_FAMILY = object()

    @bh.utils.set_family(NEW_FAMILY)
    class MyHist(bh.Histogram):
        pass

    @bh.utils.set_family(NEW_FAMILY)
    class MyRegular(bh.axis.Regular):
        __slots__ = ()

    @bh.utils.set_family(NEW_FAMILY)
    class MyIntStorage(bh.storage.Int64):
        pass

    @bh.utils.set_family(NEW_FAMILY)
    class MyPowTransform(bh.axis.transform.Pow):
        pass

    h = MyHist(MyRegular(10, 0, 2, transform=MyPowTransform(2)), storage=MyIntStorage())

    assert type(h) == MyHist
    assert h._storage_type == MyIntStorage
    assert type(h.axes[0]) == MyRegular
    assert type(h.axes[0].transform) == MyPowTransform
