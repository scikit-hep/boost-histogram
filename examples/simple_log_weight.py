#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import boost_histogram as bh

# make 1-d histogram with 5 logarithmic bins from 1e0 to 1e5
h = bh.Histogram(
    bh.axis.Regular(5, 1e0, 1e5, metadata="x", transform=bh.axis.transform.log),
    storage=bh.storage.Weight(),
)

# fill histogram with numbers
x = (2e0, 2e1, 2e2, 2e3, 2e4)
h.fill(x, weight=4)  # increment bin counter by 4

# iterate over bins and access bin counter
for idx, (lower, upper) in enumerate(h.axes[0]):
    val = h[idx]
    print(
        "bin {} [{:g}, {:g}): {} +/- {}".format(
            idx, lower, upper, val.value, val.variance ** 0.5
        )
    )

# under- and overflow bin
lo, up = h.axes[0][bh.underflow]
print(
    "underflow [{:g}, {:g}): {} +/- {}".format(
        lo, up, h[bh.underflow].value, h[bh.underflow].variance ** 0.5
    )
)
lo, up = h.axes[0][bh.overflow]
print(
    "overflow  [{:g}, {:g}): {} +/- {}".format(
        lo, up, h[bh.overflow].value, h[bh.overflow].variance ** 0.5
    )
)

# prints:
# bin 0 x in [1.0, 10.0): 4.0 +/- 4.0
# bin 1 x in [10.0, 100.0): 4.0 +/- 4.0
# bin 2 x in [100.0, 1000.0): 4.0 +/- 4.0
# bin 3 x in [1000.0, 10000.0): 4.0 +/- 4.0
# bin 4 x in [10000.0, 100000.0): 4.0 +/- 4.0
# underflow [0.0, 1.0): 0.0 +/- 0.0
# overflow  [100000.0, inf): 0.0 +/- 0.0
