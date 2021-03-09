# bh.sum is just the Python sum, so from boost_histogram import * is safe (but
# not recommended)
from builtins import sum
from typing import TypeVar, Union

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
        from copy import copy

        other = copy(self)
        other.offset += offset
        return other

    def __sub__(self: T, offset: int) -> T:
        from copy import copy

        other = copy(self)
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

    def __init__(self, value: Union[str, float], offset: int = 0) -> None:
        super().__init__(offset)
        self.value = value

    def _print_self_(self) -> str:
        return f"({self.value})"

    def __call__(self, axis: AxisLike) -> int:
        return axis.index(self.value) + self.offset


class Underflow(Locator):
    __slots__ = ()
    NAME = "underflow"

    def __call__(self, axis: AxisLike) -> int:
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

    def __call__(self, axis: AxisLike) -> int:
        return self.value


class rebin:
    __slots__ = ("factor",)

    def __init__(self, value: int) -> None:
        self.factor = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.factor})"

    # TODO: Add __call__ to support UHI
