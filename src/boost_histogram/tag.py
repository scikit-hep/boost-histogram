# bh.sum is just the Python sum, so from boost_histogram import * is safe (but
# not recommended)
from __future__ import annotations

import copy
from builtins import sum
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    from uhi.typing.plottable import PlottableAxis

from ._compat.typing import Self
from .typing import AxisLike

__all__ = ("Locator", "Slicer", "at", "loc", "overflow", "rebin", "sum", "underflow")


def Slicer() -> np.lib._index_tricks_impl.IndexExpression:
    """
    It is encouraged to use "np.s_" directly instead of this function:

        h[{0: np.s_[::bh.rebin(2)]}]   # rebin axis 0 by two

    This is provided for backward compatibility.
    """
    return np.s_


T = TypeVar("T", bound="Locator")


class Locator:
    __slots__ = ("offset",)
    NAME = ""

    def __init__(self, offset: int = 0) -> None:
        if not isinstance(offset, int):
            raise ValueError("The offset must be an integer")

        self.offset = offset

    def __add__(self, offset: int) -> Self:
        other = copy.copy(self)
        other.offset += offset
        return other

    def __sub__(self, offset: int) -> Self:
        other = copy.copy(self)
        other.offset -= offset
        return other

    def _print_self_(self) -> str:
        return ""

    def __repr__(self) -> str:
        s = self.NAME or self.__class__.__name__
        s += self._print_self_()
        if self.offset != 0:
            s += " + " if self.offset > 0 else " - "
            s += str(abs(self.offset))
        return s


class loc(Locator):
    __slots__ = ("value",)

    def __init__(self, value: str | float, offset: int = 0) -> None:
        super().__init__(offset)
        self.value = value

    def _print_self_(self) -> str:
        return f"({self.value})"

    def __call__(self, axis: AxisLike) -> int:
        return axis.index(self.value) + self.offset


class Underflow(Locator):
    __slots__ = ()
    NAME = "underflow"

    def __call__(self, axis: AxisLike) -> int:  # noqa: ARG002
        return -1 + self.offset


underflow = Underflow()


class Overflow(Locator):
    __slots__ = ()
    NAME = "overflow"

    def __call__(self, axis: AxisLike) -> int:
        return len(axis) + self.offset


overflow = Overflow()


class at:
    __slots__ = ("value",)

    def __init__(self, value: int) -> None:
        self.value = value

    def __call__(self, axis: AxisLike) -> int:  # noqa: ARG002
        return self.value


class rebin:
    __slots__ = (
        "axis",
        "edges",
        "factor",
        "groups",
    )

    def __init__(
        self,
        factor_or_axis: int | PlottableAxis | None = None,
        /,
        *,
        factor: int | None = None,
        groups: Sequence[int] | None = None,
        edges: Sequence[int | float] | None = None,
        axis: PlottableAxis | None = None,
    ) -> None:
        if isinstance(factor_or_axis, int):
            factor = factor_or_axis
        elif factor_or_axis is not None:
            axis = factor_or_axis

        total_args = sum(i is not None for i in [factor, groups, edges])
        if total_args != 1 and axis is None:
            raise ValueError("Exactly one argument should be provided")

        self.groups = groups
        self.edges = edges
        self.axis = axis
        self.factor = factor

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}"
        args: dict[str, int | Sequence[int | float] | PlottableAxis | None] = {
            "factor": self.factor,
            "groups": self.groups,
            "edges": self.edges,
            "axis": self.axis,
        }
        for k, v in args.items():
            if v is not None:
                return_str = f"{repr_str}({k}={v})"
                break
        return return_str

    # Note: this preserves the input type of `self.axis`, so is safe within a
    # single UHI library, but not cross-library. Returns None for the axis if
    # an axis is not provided, the caller should make an axis if that's the
    # case.
    def axis_mapping(
        self, axis: PlottableAxis
    ) -> tuple[Sequence[int], PlottableAxis | None]:
        return (self.group_mapping(axis), self.axis)

    def group_mapping(self, axis: PlottableAxis) -> Sequence[int]:
        if self.groups is not None:
            if sum(self.groups) != len(axis):
                msg = f"The sum of the groups ({sum(self.groups)}) must be equal to the number of bins in the axis ({len(axis)})"
                raise ValueError(msg)
            return self.groups
        if self.factor is not None:
            return [self.factor] * len(axis)
        if self.edges is not None or self.axis is not None:
            newedges = None
            if self.edges is not None:
                newedges = self.edges
            elif self.axis is not None and hasattr(self.axis, "edges"):
                newedges = self.axis.edges

            if newedges is not None and hasattr(axis, "edges"):
                if newedges[0] != axis.edges[0]:
                    msg = "Edges must start at first bin"
                    raise ValueError(msg)
                if newedges[-1] != axis.edges[-1]:
                    msg = "Edges must end at last bin"
                    raise ValueError(msg)
                matched_ixes = [np.abs(axis.edges - edge).argmin() for edge in newedges]
                missing_edges = [
                    edge
                    for ix, edge in zip(matched_ixes, newedges)
                    if not np.isclose(axis.edges[ix], edge)
                ]
                if missing_edges:
                    missing_edges_repr = ", ".join(map(str, missing_edges))
                    msg = f"Edge(s) {missing_edges_repr} not found in axis"
                    raise ValueError(msg)
                return [
                    int(ix - matched_ixes[i]) for i, ix in enumerate(matched_ixes[1:])
                ]
        raise ValueError("No rebinning factor or groups provided")
