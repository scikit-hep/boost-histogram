#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import boost_histogram as bh

# Make 1-d histogram with 5 logarithmic bins from 1e0 to 1e5
h = bh.Histogram(
    bh.axis.Regular(5, 1e0, 1e5, metadata="x", transform=bh.axis.transform.log),
    storage=bh.storage.Weight(),
)

# Fill histogram with numbers
x = (2e0, 2e1, 2e2, 2e3, 2e4)

# Doing this several times so the variance is more interesting
h.fill(x, weight=2)
h.fill(x, weight=2)
h.fill(x, weight=2)
h.fill(x, weight=2)

# Iterate over bins and access bin counter
for idx, (lower, upper) in enumerate(h.axes[0]):
    val = h[idx]
    print(f"bin {idx} in [{lower:g}, {upper:g}): {val.value} +/- {val.variance**.5}")
