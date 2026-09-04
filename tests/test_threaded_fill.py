from __future__ import annotations

import sys
import threading

import numpy as np
import pytest
from pytest import approx

import boost_histogram as bh

if sys.platform.startswith("emscripten"):
    pytest.skip(allow_module_level=True)


def fillit(hist, *args, **kwargs):
    return hist.reset().fill(*args, **kwargs)


@pytest.mark.benchmark(group="threaded-fill-1d")
@pytest.mark.parametrize("threads", [1, 4], ids=lambda x: f"threads={x}")
# @pytest.mark.parametrize("atomic", [True, False], ids=["atomic", "double"])
@pytest.mark.parametrize(
    "storage", [bh.storage.AtomicInt64, bh.storage.Double, bh.storage.Int64]
)
def test_threads(benchmark, threads, storage):
    axes = [bh.axis.Regular(100, 0, 1)]
    hist_linear = bh.Histogram(*axes, storage=storage())
    hist_atomic = hist_linear.copy()

    vals = np.random.rand(1000000)
    hist_linear.fill(vals)
    hist_result = benchmark(fillit, hist_atomic, vals, threads=threads)

    assert np.asarray(hist_linear) == approx(np.asarray(hist_result))


@pytest.mark.parametrize("threads", [1, 4, 7], ids=lambda x: f"threads={x}")
@pytest.mark.parametrize(
    "storage", [bh.storage.AtomicInt64, bh.storage.Double, bh.storage.Int64]
)
def test_threaded_builtin(threads, storage):
    axes = [bh.axis.Regular(1000, 0, 1)]
    hist_atomic1 = bh.Histogram(*axes, storage=storage())
    hist_atomic2 = hist_atomic1.copy()

    vals = np.random.rand(10003)

    hist_atomic1.fill(vals)
    hist_atomic2.fill(vals, threads=threads)

    assert np.asarray(hist_atomic1) == approx(np.asarray(hist_atomic2))


@pytest.mark.parametrize("threads", [1, 4, 7], ids=lambda x: f"threads={x}")
def test_threaded_numpy(threads):
    vals = np.random.rand(10003)

    hist_1, _ = bh.numpy.histogram(vals)
    hist_2, _ = bh.numpy.histogram(vals, threads=threads)

    assert np.asarray(hist_1) == approx(np.asarray(hist_2))


@pytest.mark.parametrize("threads", [1, 4, 7], ids=lambda x: f"threads={x}")
def test_threaded_weights(threads):
    x, y, weights = np.random.rand(3, 10003)

    hist_1 = bh.Histogram(bh.axis.Regular(10, 0, 1), bh.axis.Regular(10, 0, 1))
    hist_2 = hist_1.copy()

    hist_1.fill(x, y, weight=weights)
    hist_2.fill(x, y, weight=weights, threads=threads)

    assert hist_1.view() == approx(hist_2.view())


@pytest.mark.parametrize("threads", [1, 4, 7], ids=lambda x: f"threads={x}")
def test_threaded_weight_storage(threads):
    x, y, weights = np.random.rand(3, 10003)

    hist_1 = bh.Histogram(
        bh.axis.Regular(10, 0, 1),
        bh.axis.Regular(10, 0, 1),
        storage=bh.storage.Weight(),
    )
    hist_2 = hist_1.copy()

    hist_1.fill(x, y, weight=weights)
    hist_2.fill(x, y, weight=weights, threads=threads)

    assert hist_1.view().value == approx(hist_2.view().value)
    assert hist_1.view().variance == approx(hist_2.view().variance)


@pytest.mark.parametrize("threads", [2, 4, 7], ids=lambda x: f"threads={x}")
def test_threaded_scalar_broadcast(threads):
    # Issue #1143 (B5): scalar positional args used to crash np.array_split
    hist_1 = bh.Histogram(bh.axis.Regular(10, 0, 1), bh.axis.Regular(10, 0, 1))
    hist_2 = hist_1.copy()

    y = np.random.rand(13)
    hist_1.fill(0.5, y)
    hist_2.fill(0.5, y, threads=threads)

    assert hist_1.view(flow=True) == approx(hist_2.view(flow=True))


@pytest.mark.parametrize("threads", [2, 4, 7], ids=lambda x: f"threads={x}")
def test_threaded_scalar_weight(threads):
    # Issue #1143 (B5): 0-d array weights are not np.isscalar but must broadcast
    x = np.random.rand(13)

    hist_1 = bh.Histogram(bh.axis.Regular(10, 0, 1), storage=bh.storage.Weight())
    hist_2 = hist_1.copy()

    hist_1.fill(x, weight=np.array(2.0))
    hist_2.fill(x, weight=np.array(2.0), threads=threads)

    assert hist_1.view().value == approx(hist_2.view().value)
    assert hist_1.view().variance == approx(hist_2.view().variance)


@pytest.mark.parametrize("threads", [2, 4])
def test_threaded_all_scalar(threads):
    # All-scalar fills must fill exactly once, not once per thread
    hist = bh.Histogram(bh.axis.Regular(10, 0, 1))
    hist.fill(0.5, threads=threads)
    assert hist.sum() == 1


def test_no_profile():
    hist = bh.Histogram(bh.axis.Regular(10, 0, 1), storage=bh.storage.Mean())
    hist.fill([1, 1], sample=[1, 1])
    with pytest.raises(RuntimeError):
        hist.fill([1, 1], sample=[1, 1], threads=2)


def test_no_weighted_profile():
    hist = bh.Histogram(bh.axis.Regular(10, 0, 1), storage=bh.storage.WeightedMean())
    hist.fill([1, 1], sample=[1, 1])
    with pytest.raises(RuntimeError):
        hist.fill([1, 1], sample=[1, 1], threads=2)


# Filling then summing produces different results for means
# @pytest.mark.parametrize("threads", [2, 4, 7], ids=lambda x: "threads={0}".format(x))
# def test_threaded_samples(threads):
#     x, y, weights = np.random.rand(3, 10003)
#     samples = np.random.randint(1, 10, size=10003)
#
#     hist_1 = bh.Histogram(
#             bh.axis.Regular(10,0,1),
#             bh.axis.Regular(10,0,1),
#             storage=bh.storage.WeightedMean())
#     hist_2 = hist_1.copy()
#
#     hist_1.fill(x, y, sample=samples, weight=weights)
#     hist_2.fill(x, y, sample=samples, weight=weights, threads=threads)
#
#     assert_almost_equal(hist_1.view().value, hist_2.view().value)
#     assert_almost_equal(hist_1.view().variance, hist_2.view().variance)
#     assert_almost_equal(hist_1.view().sum_of_weights, hist_2.view().sum_of_weights)


@pytest.mark.parametrize("threads", [2, 4])
def test_threaded_continuous_growth_raises(threads):
    # A continuous growth axis cannot merge across threads yet; the worker
    # error must reach the caller instead of losing the data silently
    data = np.linspace(0, 10, 1000)

    hist = bh.Histogram(bh.axis.Regular(4, 0, 1, growth=True))
    # Match only the start of Boost's message, which spells the last word its
    # own way
    with pytest.raises(ValueError, match="axes not"):
        hist.fill(data, threads=threads)


@pytest.mark.parametrize("threads", [2, 4])
def test_threaded_category_growth(threads):
    values = ["a", "b", "c"] * 100

    hist_1 = bh.Histogram(bh.axis.StrCategory([], growth=True))
    hist_1.fill(values)

    hist_2 = bh.Histogram(bh.axis.StrCategory([], growth=True))
    hist_2.fill(values, threads=threads)

    assert hist_2.sum() == 300
    assert list(hist_2.axes[0]) == list(hist_1.axes[0])
    assert hist_2 == hist_1


@pytest.mark.parametrize("threads", [2, 4])
def test_threaded_int_category_growth(threads):
    values = np.array([1, 2, 3] * 100)

    hist_1 = bh.Histogram(bh.axis.IntCategory([], growth=True))
    hist_1.fill(values)

    hist_2 = bh.Histogram(bh.axis.IntCategory([], growth=True))
    hist_2.fill(values, threads=threads)

    assert hist_2.sum() == 300
    assert list(hist_2.axes[0]) == list(hist_1.axes[0])
    assert hist_2 == hist_1


@pytest.mark.parametrize("threads", [2, 4])
def test_threaded_integer_growth(threads):
    values = np.arange(300) % 7

    hist_1 = bh.Histogram(bh.axis.Integer(0, 1, growth=True))
    hist_1.fill(values)

    hist_2 = bh.Histogram(bh.axis.Integer(0, 1, growth=True))
    hist_2.fill(values, threads=threads)

    assert hist_2.sum() == 300
    assert hist_2.axes[0].size == hist_1.axes[0].size
    assert hist_2 == hist_1


@pytest.mark.parametrize("storage", [bh.storage.Double, bh.storage.Weight])
def test_threaded_whole_buffer_ops(storage):
    # Copy, reset, +=, and == release the GIL; run them from several threads
    # at once to confirm the results stay right and nothing deadlocks on the
    # metadata dicts the axes carry.
    base = bh.Histogram(
        bh.axis.Regular(1000, 0, 1, metadata={"a": 1}),
        bh.axis.Integer(0, 200, metadata="x"),
        storage=storage(),
    )
    base.fill(np.random.rand(10007), np.random.randint(0, 200, 10007))

    total = base.copy()
    total.reset()
    lock = threading.Lock()
    errors = []

    def work():
        nonlocal total
        try:
            local = base.copy()
            assert local == base
            with lock:
                total += local
        except Exception as err:  # noqa: BLE001
            errors.append(err)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert total.values(flow=True) == approx(base.values(flow=True) * 8)


def test_inplace_op_returns_same_object():
    h1 = bh.Histogram(bh.axis.Regular(10, 0, 1))
    h2 = h1.copy()
    assert h1._hist.__iadd__(h2._hist) is h1._hist


def test_threaded_growth_copy_race():
    # The per-thread copy and the merge both release the GIL; a copy taken
    # while another worker merged used to read a half-grown histogram
    values = ["a", "b", "c"] * 3000

    for _ in range(50):
        hist = bh.Histogram(bh.axis.StrCategory([], growth=True))
        hist.fill(values, threads=8)
        assert hist.sum() == 9000
