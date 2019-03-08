import hist

def test_regular_histogram_basic():

    h = hist.regular_histogram(hist.regular_axes_storage([hist.regular_axis(10,0,1)]))
    h(.2)
    assert h.at(2) == 1.0
