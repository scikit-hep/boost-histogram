import numpy as np
import pytest
from pytest import approx

import boost_histogram as bh

hypothesis = pytest.importorskip("hypothesis")
nst = pytest.importorskip("hypothesis.extra.numpy")


@hypothesis.given(
    nst.arrays(
        float,
        (4,),
        elements=dict(min_value=1, max_value=100, exclude_min=True, allow_nan=False),
    ),
    nst.arrays(
        float, (4,), elements=dict(min_value=-100, max_value=100, allow_nan=False)
    ),
    nst.arrays(
        float,
        (4,),
        elements=dict(min_value=0, max_value=100, allow_nan=False, exclude_min=True),
    ),
)
def test_variance_setting(cnt, val, var):
    h = bh.Histogram(bh.axis.Regular(4, 0, 1), storage=bh.storage.Mean())
    h[...] = np.stack([cnt, val, var * cnt], axis=-1)

    assert h.counts() == approx(cnt)
    assert h.values() == approx(val)
    assert h.variances() == approx(var)  # , abs=1e-3, rel=1e-3)
