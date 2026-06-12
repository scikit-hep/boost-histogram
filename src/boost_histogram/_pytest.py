"""
A pytest plugin that produces readable explanations when an ``assert a == b``
between two :class:`~boost_histogram.Histogram` (or axis) objects fails.

It is registered as a ``pytest11`` entry point (see ``pyproject.toml``), so it
loads automatically for any project that has ``boost-histogram`` installed. The
hook is inert for every comparison that is not a ``==`` between two histograms
or two axes, so it never interferes with unrelated assertions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .axis import Axis
from .histogram import Histogram

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["pytest_assertrepr_compare"]


def __dir__() -> list[str]:
    return __all__


# Maximum number of individual differing bins to list before truncating.
_MAX_SAMPLES = 5


def pytest_assertrepr_compare(op: str, left: object, right: object) -> list[str] | None:
    """
    pytest hook: return a custom explanation for a failing ``left op right``.

    Returning ``None`` falls back to pytest's default representation.
    """
    if op != "==":
        return None
    if isinstance(left, Histogram) and isinstance(right, Histogram):
        return _histogram_explanation(left, right)
    if isinstance(left, Axis) and isinstance(right, Axis):
        return ["Axes are not equal:", *_indent(_axis_explanation(left, right))]
    return None


def _indent(lines: list[str]) -> list[str]:
    return [f"  {line}" for line in lines]


def _histogram_explanation(left: Histogram[Any], right: Histogram[Any]) -> list[str]:
    lines = [f"{left!r} == {right!r}", "Histograms are not equal:"]

    # 1. Rank (number of axes).
    if left.ndim != right.ndim:
        lines.append(f"  ndim: {left.ndim} != {right.ndim}")
        return lines

    # 2. Axes. Value bins are only meaningful once the binning matches, so if any
    # axis differs we explain the first difference and stop.
    differing = [
        (i, la, ra)
        for i, (la, ra) in enumerate(zip(left.axes, right.axes, strict=True))
        if la != ra
    ]
    if differing:
        lines.append("  axes differ:")
        for i, la, ra in differing:
            lines.append(f"    axis {i}: {la!r} != {ra!r}")
        i, la, ra = differing[0]
        lines.append(f"  axis {i} details:")
        lines.extend(_indent(_indent(_axis_explanation(la, ra))))
        return lines

    # 3. Storage type.
    left_storage = left.storage_type.__name__
    right_storage = right.storage_type.__name__
    if left_storage != right_storage:
        lines.append(f"  storage: {left_storage} != {right_storage}")
        return lines

    # 4. Bin contents (including flow, matching ``__eq__`` semantics). Note that
    # histogram-level Python ``metadata`` is not part of ``__eq__`` (only the
    # underlying C++ histogram is compared), so it is intentionally not reported.
    lines.extend(_indent(_contents_explanation(left, right)))
    return lines


def _contents_explanation(left: Histogram[Any], right: Histogram[Any]) -> list[str]:
    lv = np.asarray(left.view(flow=True))
    rv = np.asarray(right.view(flow=True))

    if lv.shape != rv.shape:  # pragma: no cover - guarded by equal axes above
        return [f"contents shape: {lv.shape} != {rv.shape}"]

    names = lv.dtype.names
    if names is None:
        return _plain_contents(lv, rv)
    return _structured_contents(lv, rv, names)


def _plain_contents(lv: NDArray[Any], rv: NDArray[Any]) -> list[str]:
    mask = lv != rv
    count = int(np.count_nonzero(mask))
    if count == 0:  # pragma: no cover - only reached on a real difference
        return ["contents differ"]

    lines = [f"contents: {count} of {lv.size} bins differ (counts include flow)"]
    indices = np.argwhere(mask)
    for idx in indices[:_MAX_SAMPLES]:
        key = tuple(int(i) for i in idx)
        lines.append(f"  bin {key}: {lv[tuple(idx)]} != {rv[tuple(idx)]}")
    if count > _MAX_SAMPLES:
        lines.append(f"  ... and {count - _MAX_SAMPLES} more")
    lines.append(f"sum (no flow): {lv.sum()} vs {rv.sum()}")
    return lines


def _structured_contents(
    lv: NDArray[Any], rv: NDArray[Any], names: tuple[str, ...]
) -> list[str]:
    lines = ["contents differ (counts include flow):"]
    for name in names:
        field_mask = lv[name] != rv[name]
        count = int(np.count_nonzero(field_mask))
        if count:
            lines.append(f"  field {name!r}: {count} of {lv.size} bins differ")
    if len(lines) == 1:  # pragma: no cover - only reached on a real difference
        lines.append("  contents differ")
    return lines


def _axis_explanation(left: Axis, right: Axis) -> list[str]:
    # 1. Different axis classes (e.g. Regular vs Variable).
    if type(left) is not type(right):
        return [f"type: {type(left).__name__} != {type(right).__name__}"]

    # 2. Constructor arguments, in the same vocabulary as the repr. This surfaces
    # differing size/start/stop/edges/categories/metadata.
    left_args = left._repr_args_()
    right_args = right._repr_args_()
    if left_args != right_args:
        lines = ["arguments differ:"]
        for la, ra in zip(left_args, right_args, strict=False):
            if la != ra:
                lines.append(f"  {la} != {ra}")
        # Trailing args present on only one side (different arity).
        for extra in left_args[len(right_args) :]:
            lines.append(f"  {extra} != <missing>")
        for extra in right_args[len(left_args) :]:
            lines.append(f"  <missing> != {extra}")
        return lines

    # 3. Fallback: reprs match but the objects still differ. Compare the edges and
    # metadata directly.
    lines = []
    left_edges = getattr(left, "edges", None)
    right_edges = getattr(right, "edges", None)
    if (
        left_edges is not None
        and right_edges is not None
        and not np.array_equal(left_edges, right_edges)
    ):
        lines.append("edges differ")
    if left.metadata != right.metadata:
        lines.append(f"metadata: {left.metadata!r} != {right.metadata!r}")
    if not lines:
        lines.append(f"{left!r} != {right!r}")
    return lines
