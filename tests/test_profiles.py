from pytest import approx
import boost_histogram as bh


def test_mean_hist():

    h = bh.histogram(bh.axis.regular(3, 0, 1), storage=bh.storage.mean)

    h.fill(0.10, sample=[2.5])
    h.fill(0.25, sample=[3.5])
    h.fill(0.45, sample=[1.2])
    h.fill(0.51, sample=[3.4])
    h.fill(0.81, sample=[1.3])
    h.fill(0.86, sample=[1.9])

    results = (
        dict(bin=0, count=2, value=3.0, variance=0.5),
        dict(bin=1, count=2, value=2.3, variance=2.42),
        dict(bin=2, count=2, value=1.6, variance=0.18),
    )

    for res, x in zip(results, bh.indexed(h)):
        ind, = x.indices()

        assert res["bin"] == ind
        assert res["count"] == x.content.count
        assert approx(res["value"]) == x.content.value
        assert approx(res["variance"]) == x.content.variance
