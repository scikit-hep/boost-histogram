import boost_histogram as bh

from concurrent.futures import ThreadPoolExecutor
from functools import partial, reduce
from operator import add


def mk_and_fill(axes, vals):
    hist = bh.Histogram(*axes)
    hist.fill(vals)
    return hist


def thread_fill(axes, threads, vals):
    with ThreadPoolExecutor(threads) as pool:
        sz = len(vals) // threads
        mk_and_fill_th = partial(mk_and_fill, axes)
        results = pool.map(
            mk_and_fill_th, [vals[i * sz : (i + 1) * sz] for i in range(threads)]
        )

    return reduce(add, results)


def atomic_fill(axes, threads, vals):
    hist = bh.Histogram(*axes, storage=bh.storage.AtomicInt64())
    with ThreadPoolExecutor(threads) as pool:
        sz = len(vals) // threads
        pool.map(hist.fill, [vals[i * sz : (i + 1) * sz] for i in range(threads)])
    return hist


def classic_fill(axes, threads, vals):
    hist = bh.Histogram(*axes)
    hist.fill(vals)
    return hist
