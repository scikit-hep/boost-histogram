import numpy as np
import pytest

import boost_histogram as bh


@pytest.mark.benchmark(group="IntCategory")
@pytest.mark.parametrize("dtype", ("i", tuple))
def test_IntCategory(benchmark, growth, dtype):
    np.random.seed(42)
    values = np.random.choice(np.arange(5), size=[100000])
    h = bh.Histogram(
        bh.axis.IntCategory([] if growth else np.arange(4), growth=growth),
        storage=bh.storage.Double(),
    )

    def run(h, data):
        h.fill(data)

    benchmark(run, h, tuple(values) if dtype is tuple else values.astype(dtype))


@pytest.mark.benchmark(group="StrCategory")
@pytest.mark.parametrize("dtype", ("S", "U", "O", tuple))
def test_StrCategory(benchmark, growth, dtype):
    np.random.seed(42)
    values = np.random.choice(["A", "B", "C", "D", "E"], size=[100000])
    h = bh.Histogram(
        bh.axis.StrCategory([] if growth else ["A", "B", "C", "D"], growth=growth),
        storage=bh.storage.Double(),
    )

    def run(h, data):
        h.fill(data)

    benchmark(run, h, tuple(values) if dtype is tuple else values.astype(dtype))


@pytest.mark.benchmark(group="Pick")
def test_pick_only(benchmark):

    h = bh.Histogram(
        bh.axis.StrCategory([str(i) for i in range(32)]),
        bh.axis.StrCategory([str(i) for i in range(32)]),
        bh.axis.StrCategory([str(i) for i in range(32)]),
        bh.axis.Regular(32, 0, 320),
    )

    h[...] = 1.0

    def run(h):
        return h[bh.loc("13"), bh.loc("13"), bh.loc("13"), :].view()

    benchmark(run, h)


@pytest.mark.benchmark(group="Pick")
def test_pick_and_slice(benchmark):

    h = bh.Histogram(
        bh.axis.StrCategory([str(i) for i in range(32)]),
        bh.axis.StrCategory([str(i) for i in range(32)]),
        bh.axis.StrCategory([str(i) for i in range(32)]),
        bh.axis.Regular(32, 0, 320),
    )

    h[...] = 1.0

    def run(h):
        return h[3:29, bh.loc("13"), bh.loc("13"), :].view()

    benchmark(run, h)
