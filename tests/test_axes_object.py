import numpy as np
import pytest

import boost_histogram as bh


@pytest.fixture
def h():
    return bh.Histogram(
        bh.axis.Regular(10, 0, 10, metadata=2),
        bh.axis.Integer(0, 5, metadata="hi"),
        bh.axis.StrCategory(["HI", "HO"]),
    )


def test_axes_basics(h):
    h = bh.Histogram(
        bh.axis.Regular(10, 0, 10, metadata=2),
        bh.axis.Integer(0, 5, metadata="hi"),
        bh.axis.StrCategory(["HI", "HO"]),
    )

    assert h.axes.bin(1, 2, 0) == ((1.0, 2.0), 2, "HI")
    assert h.axes.value(0, 0, 0) == (0.0, 0, "HI")
    assert h.axes.index(2, 3, "HO") == (2, 3, 1)

    assert h.axes.size == (10, 5, 2)
    assert h.axes.extent == (12, 7, 3)
    assert h.axes.metadata == (2, "hi", None)

    h.axes.metadata = None, 3, "bye"

    assert h.axes.metadata == (None, 3, "bye")


def test_axes_centers(h):
    centers = h.axes.centers
    answers = np.ogrid[0.5:10, 0.5:5, 0.5:2]
    full_answers = np.mgrid[0.5:10, 0.5:5, 0.5:2]

    for i in range(3):
        np.testing.assert_allclose(centers.broadcast()[i], full_answers[i])
        np.testing.assert_allclose(centers[i], answers[i])
        np.testing.assert_allclose(centers.T[i], answers[i].T)
        np.testing.assert_allclose(centers.flatten()[i], answers[i].flatten())
        np.testing.assert_allclose(h.axes[i].centers, answers[i].ravel())


def test_axes_edges(h):
    edges = h.axes.edges
    answers = np.ogrid[0:11, 0:6, 0:3]
    full_answers = np.mgrid[0:11, 0:6, 0:3]

    for i in range(3):
        np.testing.assert_allclose(edges.broadcast()[i], full_answers[i])
        np.testing.assert_allclose(edges[i], answers[i])
        np.testing.assert_allclose(edges.T[i], answers[i].T)
        np.testing.assert_allclose(edges.ravel()[i], answers[i].ravel())
        np.testing.assert_allclose(h.axes[i].edges, answers[i].ravel())


def test_axes_widths(h):
    widths = h.axes.widths
    answers = np.ogrid[1:1:10j, 1:1:5j, 1:1:2j]
    full_answers = np.mgrid[1:1:10j, 1:1:5j, 1:1:2j]

    for i in range(3):
        np.testing.assert_allclose(widths.broadcast()[i], full_answers[i])
        np.testing.assert_allclose(widths[i], answers[i])
        np.testing.assert_allclose(widths.T[i], answers[i].T)
        np.testing.assert_allclose(widths.ravel()[i], answers[i].ravel())
        np.testing.assert_allclose(h.axes[i].widths, answers[i].ravel())


def test_axis_misconstuct():
    inp = [bh.axis.Regular(12, 0, 1)]
    ok = bh.axis.AxesTuple(inp)
    assert ok[0] == inp[0]

    with pytest.raises(TypeError):
        bh.axis.AxesTuple(inp[0])
