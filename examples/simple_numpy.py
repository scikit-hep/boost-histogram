#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

import boost_histogram as bh

# Create 2d-histogram with two axes with 10 equidistant bins from -3 to 3
h = bh.Histogram(
    bh.axis.Regular(10, -3, 3, metadata="x"), bh.axis.Regular(10, -3, 3, metadata="y")
)

# Generate some Numpy arrays with data to fill into histogram,
# in this case normal distributed random numbers in x and y
x_data = np.random.randn(1000)
y_data = 0.5 * np.random.randn(1000)

# Fill histogram with numpy arrays, this is very fast
h.fill(x_data, y_data)

# Get representations of the bin edges as Numpy arrays
x = h.axes[0].edges
y = h.axes[1].edges

# Creates a view of the counts (no copy involved)
count_matrix = h.view()

# Draw the count matrix
plt.pcolor(x, y, count_matrix.T)
plt.xlabel(h.axes[0].metadata)
plt.ylabel(h.axes[1].metadata)
plt.savefig("simple_numpy.png")
