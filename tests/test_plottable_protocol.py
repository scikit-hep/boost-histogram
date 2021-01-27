# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_allclose

import boost_histogram as bh

COUNTS = np.array([3, 4, 5, 6])
VALUES = np.array([1, 2, 3, 4])
VARIANCES = np.array([0.1, 0.1, 0.2, 0.2])
SUM2 = np.array([9, 8, 7, 6])  # not in protocol, so not checked


def test_plottable_histogram_mean_int():
    h = bh.Histogram(bh.axis.Integer(0, 4), storage=bh.storage.Mean())

    h[...] = np.stack([COUNTS, VALUES, VARIANCES]).T

    assert_allclose(COUNTS, h.counts())
    assert_allclose(VALUES, h.values())
    assert_allclose(VARIANCES / COUNTS, h.variances())

    assert h.kind == bh.Kind.MEAN
    assert h.kind == "MEAN"

    assert len(h.axes) == 1
    assert h.axes[0] == h.axes[0]

    assert h.axes[0][0] == 0
    assert h.axes[0][1] == 1
    assert h.axes[0][2] == 2
    assert h.axes[0][3] == 3

    assert h.axes[0].traits.discrete
    assert not h.axes[0].traits.circular


def test_plottible_histogram_weight_reg():
    h = bh.Histogram(bh.axis.Regular(4, 0, 4), storage=bh.storage.Weight())

    h[...] = np.stack([VALUES, VARIANCES]).T

    assert_allclose(VALUES, h.counts())
    assert_allclose(VALUES, h.values())
    assert_allclose(VARIANCES, h.variances())

    assert h.kind == bh.Kind.COUNT
    assert h.kind == "COUNT"

    assert len(h.axes) == 1
    assert h.axes[0] == h.axes[0]

    assert_allclose(h.axes[0][0], (0, 1))
    assert_allclose(h.axes[0][1], (1, 2))
    assert_allclose(h.axes[0][2], (2, 3))
    assert_allclose(h.axes[0][3], (3, 4))

    assert not h.axes[0].traits.discrete
    assert not h.axes[0].traits.circular


def test_plottible_histogram_simple_var():
    h = bh.Histogram(bh.axis.Variable([0, 1, 2, 3, 4]))

    # Setting directly or filling with weight should cause variances to
    # be None.  Manipulations directly on the view, however, are up to the user
    # - once you've asked for a view of the memory, you should know what your
    # are doing. At least if you change it inplace.
    h.view()[...] = VALUES

    assert_allclose(VALUES, h.counts())
    assert_allclose(VALUES, h.values())
    assert_allclose(VALUES, h.variances())

    assert h.kind == bh.Kind.COUNT
    assert h.kind == "COUNT"

    assert len(h.axes) == 1
    assert h.axes[0] == h.axes[0]

    assert_allclose(h.axes[0][0], (0, 1))
    assert_allclose(h.axes[0][1], (1, 2))
    assert_allclose(h.axes[0][2], (2, 3))
    assert_allclose(h.axes[0][3], (3, 4))

    assert not h.axes[0].traits.discrete
    assert not h.axes[0].traits.circular

    h.fill([1], weight=0)

    assert_allclose(VALUES, h.counts())
    assert_allclose(VALUES, h.values())
    assert h.variances() is None


def test_plottible_histogram_simple_var_invalidate_inplace():
    h = bh.Histogram(bh.axis.Variable([0, 1, 2, 3, 4]))
    h.view()[...] = VALUES

    h2 = h * 1

    assert_allclose(VALUES, h.counts())
    assert_allclose(VALUES, h.values())
    assert_allclose(VALUES, h.variances())

    assert_allclose(VALUES, h2.counts())
    assert_allclose(VALUES, h2.values())
    assert h2.variances() is None

    h *= 1

    assert_allclose(VALUES, h.counts())
    assert_allclose(VALUES, h.values())
    assert h.variances() is None
