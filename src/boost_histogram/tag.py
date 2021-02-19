# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import sys

# bh.sum is just the Python sum, so from boost_histogram import * is safe (but
# not recommended)
if sys.version_info < (3, 0):
    from __builtin__ import sum
else:
    from builtins import sum

del absolute_import, division, print_function

__all__ = ("Slicer", "Locator", "at", "loc", "overflow", "underflow", "rebin", "sum")


class Slicer(object):
    """
    This is a simple class to make slicing inside dictionaries simpler.
    This is how it should be used:

        s = bh.tag.Slicer()

        h[{0: s[::bh.rebin(2)]}]   # rebin axis 0 by two

    """

    def __getitem__(self, item):
        # type: (slice) -> slice
        return item


class Locator(object):
    __slots__ = ("offset",)
    NAME = ""

    def __init__(self, offset=0):
        # type: (int) -> None
        if not isinstance(offset, int):
            raise ValueError("The offset must be an integer")

        self.offset = offset

    def __add__(self, offset):
        # type: (int) -> Locator
        from copy import copy

        other = copy(self)
        other.offset += offset
        return other

    def __sub__(self, offset):
        # type: (int) -> Locator
        from copy import copy

        other = copy(self)
        other.offset -= offset
        return other

    def _print_self_(self):
        # type: () -> str
        return ""

    def __repr__(self):
        # type: () -> str
        s = self.NAME or self.__class__.__name__
        s += self._print_self_()
        if self.offset != 0:
            s += " + " if self.offset > 0 else " - "
            s += str(abs(self.offset))
        return s


class loc(Locator):
    __slots__ = ("value",)

    def __init__(self, value, offset=0):
        # type: (float, int) -> None
        super(loc, self).__init__(offset)
        self.value = value

    def _print_self_(self):
        # type: () -> str
        return "({})".format(self.value)

    def __call__(self, axis):
        return axis.index(self.value) + self.offset


class Underflow(Locator):
    __slots__ = ()
    NAME = "underflow"

    def __call__(self, axis):
        return -1 + self.offset


underflow = Underflow()


class Overflow(Locator):
    __slots__ = ()
    NAME = "overflow"

    def __call__(self, axis):
        return len(axis) + self.offset


overflow = Overflow()


class at(object):
    __slots__ = ("value",)

    def __init__(self, value):
        # type: (float) -> None
        self.value = value

    def __call__(self, axis):
        return self.value


class rebin(object):
    __slots__ = ("factor",)

    def __init__(self, value):
        # type: (int) -> None
        self.factor = value

    def __repr__(self):
        # type: () -> str
        return "{self.__class__.__name__}({self.factor})".format(self=self)

    # TODO: Add __call__ to support UHI
