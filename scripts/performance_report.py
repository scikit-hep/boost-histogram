# -*- coding: utf-8 -*-
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
vals = np.random.normal(size=[10_000_000]).astype(np.float32)
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
vals = np.random.normal(size=[2, 1_000_000]).astype(np.float64)
"""


def timer(setup, statement, n=10):
    t = timeit(statement, setup, number=n)
    t = (t * 1000) / n
    return t


def print_timer(setup, statement, name, storage, fill, flow, base=None, n=10):
    time = timer(setup, statement.format(fill=fill), n)
    speedup = 1 if base is None else base / time
    print(
        table_line.format(
            name=name, storage=storage, fill=fill, flow=flow, time=time, speedup=speedup
        )
    )
    return time


c = multiprocessing.cpu_count()
counts = [c // 2 ** x for x in reversed(range(int(math.log2(c) + 1)))]

table_header = "| Type  | Storage | Fill | Flow |    Time   | Speedup |"
table_spacer = "|-------|---------|------|------|-----------|---------|"
table_line = (
    "|{name:^7}|{storage:^9}|{fill:^6}|{flow:^6}|{time:>7.4g} ms |{speedup:>7.2g}x |"
)

print()
print("### 1D, 10 runs each")
print()
print(table_header)
print(table_spacer)


base = print_timer(
    setup_1d,
    "h, _ = np.histogram(vals, bins=bins, range=ranges)",
    name="Numpy",
    storage="uint64",
    fill="",
    flow="no",
)


print_timer(
    setup_1d,
    "hist = bh.hist._any_int([bh.axis._regular_uoflow(bins, *ranges)]); hist.fill(vals)",
    name="Any",
    storage="int",
    fill="",
    flow="yes",
    base=base,
)

print_timer(
    setup_1d,
    "hist = bh.hist._any_int([bh.axis._regular_noflow(bins, *ranges)]); hist.fill(vals)",
    name="Any",
    storage="int",
    fill="",
    flow="no",
    base=base,
)


print_timer(
    setup_1d,
    "hist = bh.hist.regular_int([bh.axis._regular_uoflow(bins, *ranges)]); hist.fill(vals)",
    name="Regular",
    storage="int",
    fill="",
    flow="yes",
    base=base,
)

print_timer(
    setup_1d,
    "hist = bh.hist._regular_noflow_int([bh.axis._regular_noflow(bins, *ranges)]); hist.fill(vals)",
    name="Regular",
    storage="int",
    fill="",
    flow="no",
    base=base,
)


print_timer(
    setup_1d,
    "hist = bh.hist.regular_atomic_int([bh.axis._regular_uoflow(bins, *ranges)]); hist.fill(vals)",
    name="Regular",
    storage="aint",
    fill="",
    flow="yes",
    base=base,
)

for fill in counts:
    print_timer(
        setup_1d,
        "hist = bh.hist.regular_atomic_int([bh.axis._regular_uoflow(bins, *ranges)]); hist.fill(vals, atomic={fill})",
        name="Regular",
        storage="aint",
        fill=fill,
        flow="yes",
        base=base,
    )

for fill in counts:
    print_timer(
        setup_1d,
        "hist = bh.hist.regular_int([bh.axis._regular_uoflow(bins, *ranges)]); hist.fill(vals, threads={fill})",
        name="Regular",
        storage="int",
        fill=fill,
        flow="yes",
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
    name="Numpy",
    storage="uint64",
    fill="",
    flow="no",
)

print_timer(
    setup_2d,
    "hist = bh.hist._any_int([bh.axis._regular_uoflow(bins[0], *ranges[0]), bh.axis._regular_uoflow(bins[1], *ranges[1])]); hist.fill(*vals)",
    name="Any",
    storage="int",
    fill="",
    flow="yes",
    base=base,
)

print_timer(
    setup_2d,
    "hist = bh.hist._any_int([bh.axis._regular_noflow(bins[0], *ranges[0]), bh.axis._regular_noflow(bins[1], *ranges[1])]); hist.fill(*vals)",
    name="Any",
    storage="int",
    fill="",
    flow="no",
    base=base,
)

print_timer(
    setup_2d,
    "hist = bh.hist.regular_int([bh.axis._regular_uoflow(bins[0], *ranges[0]), bh.axis._regular_uoflow(bins[1], *ranges[1])]); hist.fill(*vals)",
    name="Regular",
    storage="int",
    fill="",
    flow="yes",
    base=base,
)

print_timer(
    setup_2d,
    "hist = bh.hist._regular_noflow_int([bh.axis._regular_noflow(bins[0], *ranges[0]), bh.axis._regular_noflow(bins[1], *ranges[1])]); hist.fill(*vals)",
    name="Regular",
    storage="int",
    fill="",
    flow="no",
    base=base,
)


print_timer(
    setup_2d,
    "hist = bh.hist.regular_atomic_int([bh.axis._regular_uoflow(bins[0], *ranges[0]), bh.axis._regular_uoflow(bins[1], *ranges[1])]); hist.fill(*vals)",
    name="Regular",
    storage="aint",
    fill="",
    flow="yes",
    base=base,
)

for fill in counts:
    print_timer(
        setup_2d,
        "hist = bh.hist.regular_atomic_int([bh.axis._regular_uoflow(bins[0], *ranges[0]), bh.axis._regular_uoflow(bins[1], *ranges[1])]); hist.fill(*vals, atomic={fill})",
        name="Regular",
        storage="aint",
        fill=fill,
        flow="yes",
        base=base,
    )

for fill in counts:
    print_timer(
        setup_2d,
        "hist = bh.hist.regular_int([bh.axis._regular_uoflow(bins[0], *ranges[0]), bh.axis._regular_uoflow(bins[1], *ranges[1])]); hist.fill(*vals, threads={fill})",
        name="Regular",
        storage="int",
        fill=fill,
        flow="yes",
        base=base,
    )
