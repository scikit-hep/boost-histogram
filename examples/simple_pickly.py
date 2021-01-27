#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from pathlib import Path

import boost_histogram as bh

h1 = bh.Histogram(bh.axis.Regular(2, -1, 1))
h2 = h1.copy()

h1.fill(-0.5)
h2.fill(0.5)

# Arithmetic operators
h3 = h1 + h2
h4 = h3 * 2

print(f"{h4[0] = }, {h4[1] = }")

h4_saved = Path("h4_saved.pkl")

# Now save the histogram
with h4_saved.open("wb") as f:
    pickle.dump(h4, f, protocol=-1)

# And load
with h4_saved.open("rb") as f:
    h5 = pickle.load(f)

assert h4 == h5
print("Succeeded in pickling a histogram!")

# Delete the file to keep things tidy
h4_saved.unlink()
