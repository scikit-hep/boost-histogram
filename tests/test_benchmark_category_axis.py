# -*- coding: utf-8 -*-
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
