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

# pcolormesh requires fully broadcast arrays for ij indexing, sadly, so get the
# edges (50x1 and 1x50 arrays) and broadcast them out to 50x50.
X, Y = np.broadcast_arrays(*hist.axes.edges)

fig, ax = plt.subplots()
mesh = ax.pcolormesh(X, Y, density)
fig.colorbar(mesh)
plt.show()
