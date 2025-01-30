#!/usr/bin/env python
from __future__ import annotations

import math
import multiprocessing
from timeit import timeit

print("Welcome to boost-histogram's performance report")

setup_1d = """
import numpy as np
import boost_histogram as bh

bins=100
ranges=(-1,1)
bins = np.asarray(bins).astype(np.int64)
ranges = np.asarray(ranges).astype(np.float64)

edges = np.linspace(*ranges, bins+1)

np.random.seed(42)
vals = np.random.normal(size=[50_000_000]).astype(np.float32)
"""

setup_2d = """
import numpy as np
import boost_histogram as bh

bins=(100, 100)
ranges=((-1,1),(-1,1))
bins = np.asarray(bins).astype(np.int64)
ranges = np.asarray(ranges).astype(np.float64)

edges = (np.linspace(*ranges[0,:], bins[0]+1),
         np.linspace(*ranges[1,:], bins[1]+1))

np.random.seed(42)
vals = np.random.normal(size=[2, 5_000_000]).astype(np.float64)
"""


def timer(setup, statement, n=10):
    t = timeit(statement, setup, number=n)
    return (t * 1000) / n


def print_timer(setup, statement, name, storage, fill, flow, base=None, n=10):
    time = timer(
        setup, statement.format(name=name, storage=storage, fill=fill, flow=flow), n
    )
    speedup = 1 if base is None else base / time
    print(
        table_line.format(
            name=name,
            storage=storage,
            fill=fill,
            flow=str(flow),
            time=time,
            speedup=speedup,
        )
    )
    return time


c = multiprocessing.cpu_count()
counts = [c // 2**x for x in reversed(range(int(math.log2(c) + 1)))]

table_header = "| Type  | Storage       | Threads  | Flow |    Time   | Speedup |"
table_spacer = "|-------|---------------|----------|------|-----------|---------|"
table_line = (
    "|{name:^7}|{storage:^15}|{fill:^9}|{flow:<6}|{time:>7.4g} ms |{speedup:>7.2g}x |"
)

print()
print("### 1D, 10 runs each")
print()
print(table_header)
print(table_spacer)


base = print_timer(
    setup_1d,
    "h, _ = np.histogram(vals, bins=bins, range=ranges)",
    name="NumPy",
    storage="uint64",
    fill="",
    flow=False,
)


print_timer(
    setup_1d,
    "hist = bh.Histogram(bh.axis.{name}(bins, *ranges, underflow={flow}, overflow={flow}), storage=bh.storage.{storage}()); hist.fill(vals)",
    name="Regular",
    storage="Int64",
    fill="",
    flow=False,
    base=base,
)

print_timer(
    setup_1d,
    "hist = bh.Histogram(bh.axis.{name}(bins, *ranges, underflow={flow}, overflow={flow}), storage=bh.storage.{storage}()); hist.fill(vals)",
    name="Regular",
    storage="Int64",
    fill="",
    flow=True,
    base=base,
)

for fill in counts:
    print_timer(
        setup_1d,
        "hist = bh.Histogram(bh.axis.{name}(bins, *ranges, underflow={flow}, overflow={flow}), storage=bh.storage.{storage}()); hist.fill(vals, threads={fill})",
        name="Regular",
        storage="Int64",
        fill=fill,
        flow=True,
        base=base,
    )

for fill in counts:
    print_timer(
        setup_1d,
        "hist = bh.Histogram(bh.axis.{name}(bins, *ranges, underflow={flow}, overflow={flow}), storage=bh.storage.{storage}()); hist.fill(vals, threads={fill})",
        name="Regular",
        storage="AtomicInt64",
        fill=fill,
        flow=True,
        base=base,
    )


print()
print("### 2D, 10 runs each")
print()
print(table_header)
print(table_spacer)

base = print_timer(
    setup_2d,
    "H, *ledges = np.histogram2d(*vals, bins=bins, range=ranges)",
    name="NumPy",
    storage="uint64",
    fill="",
    flow=False,
)

print_timer(
    setup_2d,
    "hist = bh.Histogram(bh.axis.{name}(bins[1], *ranges[0], underflow={flow}, overflow={flow}), bh.axis.{name}(bins[1], *ranges[1], underflow={flow}, overflow={flow}), storage=bh.storage.{storage}()); hist.fill(*vals)",
    name="Regular",
    storage="Int64",
    fill="",
    flow=False,
    base=base,
)

for fill in counts:
    print_timer(
        setup_2d,
        "hist = bh.Histogram(bh.axis.{name}(bins[1], *ranges[0], underflow={flow}, overflow={flow}), bh.axis.{name}(bins[1], *ranges[1], underflow={flow}, overflow={flow}), storage=bh.storage.{storage}()); hist.fill(*vals, threads={fill})",
        name="Regular",
        storage="Int64",
        fill=fill,
        flow=False,
        base=base,
    )

for fill in counts:
    print_timer(
        setup_2d,
        "hist = bh.Histogram(bh.axis.{name}(bins[1], *ranges[0], underflow={flow}, overflow={flow}), bh.axis.{name}(bins[1], *ranges[1], underflow={flow}, overflow={flow}), storage=bh.storage.{storage}()); hist.fill(*vals, threads={fill})",
        name="Regular",
        storage="AtomicInt64",
        fill=fill,
        flow=False,
        base=base,
    )
