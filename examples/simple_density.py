#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
import operator

import matplotlib.pyplot as plt
import numpy as np

import boost_histogram as bh

# Make a 2D histogram
hist = bh.Histogram(bh.axis.Regular(50, -3, 3), bh.axis.Regular(50, -3, 3))

# Fill with Gaussian random values
hist.fill(np.random.normal(size=1_000_000), np.random.normal(size=1_000_000))

# Compute the areas of each bin
areas = functools.reduce(operator.mul, hist.axes.widths)

# Compute the density
density = hist.view() / hist.sum() / areas

# Make the plot
fig, ax = plt.subplots()
mesh = ax.pcolormesh(*hist.axes.edges.T, density.T)
fig.colorbar(mesh)
plt.savefig("simple_density.png")
