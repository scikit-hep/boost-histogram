from __future__ import annotations

import importlib.metadata
import sys

import pytest

import boost_histogram as bh
from boost_histogram._pytest import pytest_assertrepr_compare

pytest_plugins = ["pytester"]


def _plugin_autoloaded() -> bool:
    """True when boost-histogram is installed with its pytest11 entry point.

    In a CMake-only / in-place build the package is importable but not
    pip-installed, so there is no entry point and the auto-load test is skipped.
    """
    eps = importlib.metadata.entry_points(group="pytest11")
    return any(ep.value == "boost_histogram._pytest" for ep in eps)


def text(lines: list[str] | None) -> str:
    assert lines is not None
    return "\n".join(lines)


# --- top-level hook gating ------------------------------------------------


def test_returns_none_for_other_ops() -> None:
    h1 = bh.Histogram(bh.axis.Regular(2, 0, 1))
    h2 = bh.Histogram(bh.axis.Regular(2, 0, 1))
    assert pytest_assertrepr_compare("!=", h1, h2) is None
    assert pytest_assertrepr_compare("<", h1, h2) is None


def test_returns_none_for_non_histogram_axis() -> None:
    assert pytest_assertrepr_compare("==", 1, 2) is None
    assert (
        pytest_assertrepr_compare("==", bh.Histogram(bh.axis.Regular(2, 0, 1)), 2)
        is None
    )


# --- histogram explanations -----------------------------------------------


def test_histogram_ndim_differs() -> None:
    h1 = bh.Histogram(bh.axis.Regular(2, 0, 1))
    h2 = bh.Histogram(bh.axis.Regular(2, 0, 1), bh.axis.Regular(2, 0, 1))
    out = text(pytest_assertrepr_compare("==", h1, h2))
    assert "ndim: 1 != 2" in out


def test_histogram_axis_differs() -> None:
    h1 = bh.Histogram(bh.axis.Regular(10, 0, 1))
    h2 = bh.Histogram(bh.axis.Regular(20, 0, 1))
    out = text(pytest_assertrepr_compare("==", h1, h2))
    assert "axes differ" in out
    assert "axis 0:" in out
    # The detailed per-axis explanation is included for the first differing axis.
    assert "arguments differ" in out


def test_histogram_storage_differs() -> None:
    h1 = bh.Histogram(bh.axis.Regular(2, 0, 1), storage=bh.storage.Double())
    h2 = bh.Histogram(bh.axis.Regular(2, 0, 1), storage=bh.storage.Int64())
    out = text(pytest_assertrepr_compare("==", h1, h2))
    assert "storage: Double != Int64" in out


def test_histogram_contents_differ_plain() -> None:
    h1 = bh.Histogram(bh.axis.Regular(10, 0, 1))
    h1.fill([0.05])
    h2 = bh.Histogram(bh.axis.Regular(10, 0, 1))
    h2.fill([0.05, 0.05])
    out = text(pytest_assertrepr_compare("==", h1, h2))
    assert "1 of 12 bins differ" in out
    assert "sum (no flow): 1.0 vs 2.0" in out


def test_histogram_contents_truncated() -> None:
    h1 = bh.Histogram(bh.axis.Regular(20, 0, 1))
    h2 = bh.Histogram(bh.axis.Regular(20, 0, 1))
    h2.fill([(i + 0.5) / 20 for i in range(20)])
    out = text(pytest_assertrepr_compare("==", h1, h2))
    assert "20 of 22 bins differ" in out
    assert "and 15 more" in out


def test_histogram_contents_differ_weight() -> None:
    h1 = bh.Histogram(bh.axis.Regular(5, 0, 1), storage=bh.storage.Weight())
    h1.fill([0.5], weight=[2.0])
    h2 = bh.Histogram(bh.axis.Regular(5, 0, 1), storage=bh.storage.Weight())
    h2.fill([0.5], weight=[3.0])
    out = text(pytest_assertrepr_compare("==", h1, h2))
    assert "contents differ" in out
    assert "field 'value'" in out


# --- axis explanations ----------------------------------------------------


def test_axis_type_differs() -> None:
    a1 = bh.axis.Regular(10, 0, 1)
    a2 = bh.axis.Variable([0, 0.5, 1])
    out = text(pytest_assertrepr_compare("==", a1, a2))
    assert "type: Regular != Variable" in out


def test_axis_size_differs() -> None:
    a1 = bh.axis.Regular(10, 0, 1)
    a2 = bh.axis.Regular(20, 0, 1)
    out = text(pytest_assertrepr_compare("==", a1, a2))
    assert "arguments differ" in out
    assert "10 != 20" in out


def test_axis_edges_differ() -> None:
    a1 = bh.axis.Variable([0, 0.5, 1])
    a2 = bh.axis.Variable([0, 0.3, 1])
    out = text(pytest_assertrepr_compare("==", a1, a2))
    assert "arguments differ" in out


def test_axis_metadata_differs() -> None:
    a1 = bh.axis.Regular(10, 0, 1, metadata="a")
    a2 = bh.axis.Regular(10, 0, 1, metadata="b")
    out = text(pytest_assertrepr_compare("==", a1, a2))
    assert "metadata='a'" in out
    assert "metadata='b'" in out


# --- end-to-end through pytest (proves the entry point is wired up) -------


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="needs subprocess")
@pytest.mark.skipif(
    not _plugin_autoloaded(),
    reason="boost-histogram not installed with its pytest11 entry point (CMake-only build)",
)
def test_plugin_loaded_end_to_end(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import boost_histogram as bh

        def test_demo():
            h1 = bh.Histogram(bh.axis.Regular(10, 0, 1)); h1.fill([0.05])
            h2 = bh.Histogram(bh.axis.Regular(10, 0, 1)); h2.fill([0.05, 0.05])
            assert h1 == h2
        """
    )
    result = pytester.runpytest_subprocess("-vv", "-p", "no:cacheprovider")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*bins differ*"])
