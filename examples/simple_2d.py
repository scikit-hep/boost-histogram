#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

import boost_histogram as bh

# Create 2d-histogram with two axes with 20 equidistant bins from -3 to 3
h = bh.Histogram(
    bh.axis.Regular(20, -3, 3, metadata="x"), bh.axis.Regular(20, -3, 3, metadata="y")
)

# Generate some Numpy arrays with data to fill into histogram,
# in this case normal distributed random numbers in x and y
x = np.random.randn(1_000)
y = 0.5 * np.random.randn(1_000)

# Fill histogram with Numpy arrays, this is very fast
h.fill(x, y)

# Get numpy.histogram compatible representation of the histogram
w, x, y = h.to_numpy()

# Draw the count matrix

fig, ax = plt.subplots()
ax.pcolormesh(x, y, w.T)
ax.set_xlabel(h.axes[0].metadata)
ax.set_ylabel(h.axes[1].metadata)
ax.set_aspect("equal")
plt.savefig("simple_2d.png")
