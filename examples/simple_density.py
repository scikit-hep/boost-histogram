#!/usr/bin/env python3

import numpy as np
import boost_histogram as bh
import matplotlib.pyplot as plt

# Make a 2D histogram
hist = bh.Histogram(bh.axis.Regular(50, -3, 3), bh.axis.Regular(50, -3, 3))

# Fill with Gaussian random values
hist.fill(np.random.normal(size=1_000_000), np.random.normal(size=1_000_000))

# Compute the areas of each bin
areas = np.prod(hist.axes.widths, axis=0)

# Compute the density
density = hist.view() / hist.sum() / areas

# Get the edges
X, Y = hist.axes.edges

# Make the plot
fig, ax = plt.subplots()
mesh = ax.pcolormesh(X.T, Y.T, density.T)
fig.colorbar(mesh)
plt.savefig("simple_density.png")
