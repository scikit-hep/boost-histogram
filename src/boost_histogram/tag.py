# bh.sum is just the Python sum, so from boost_histogram import * is safe (but
# not recommended)
from __future__ import annotations

import copy
from builtins import sum
from typing import TYPE_CHECKING, Sequence, TypeVar

if TYPE_CHECKING:
    from uhi.typing.plottable import PlottableAxis

from ._internal.typing import AxisLike

__all__ = ("Slicer", "Locator", "at", "loc", "overflow", "underflow", "rebin", "sum")


class Slicer:
    """
    This is a simple class to make slicing inside dictionaries simpler.
    This is how it should be used:

        s = bh.tag.Slicer()

        h[{0: s[::bh.rebin(2)]}]   # rebin axis 0 by two

    """

    def __getitem__(self, item: slice) -> slice:
        return item


T = TypeVar("T", bound="Locator")


class Locator:
    __slots__ = ("offset",)
    NAME = ""

    def __init__(self, offset: int = 0) -> None:
        if not isinstance(offset, int):
            raise ValueError("The offset must be an integer")

        self.offset = offset

    def __add__(self: T, offset: int) -> T:
        other = copy.copy(self)
        other.offset += offset
        return other

    def __sub__(self: T, offset: int) -> T:
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
        "factor",
        "groups",
    )

    def __init__(
        self,
        factor: int | None = None,
        *,
        groups: Sequence[int] | None = None,
    ) -> None:
        if not sum(i is None for i in [factor, groups]) == 1:
            raise ValueError("Exactly one, a factor or groups should be provided")
        self.factor = factor
        self.groups = groups

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}"
        args: dict[str, int | Sequence[int] | None] = {
            "factor": self.factor,
            "groups": self.groups,
        }
        for k, v in args.items():
            if v is not None:
                return_str = f"{repr_str}({k}={v})"
                break
        return return_str

    def group_mapping(self, axis: PlottableAxis) -> Sequence[int]:
        if self.groups is not None:
            if sum(self.groups) != len(axis):
                msg = f"The sum of the groups ({sum(self.groups)}) must be equal to the number of bins in the axis ({len(axis)})"
                raise ValueError(msg)
            return self.groups
        if self.factor is not None:
            return [self.factor] * len(axis)
        raise ValueError("No rebinning factor or groups provided")
