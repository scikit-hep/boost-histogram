from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function


class loc(object):
    __slots__ = ("value", "offset")

    def __init__(self, value, offset=0):
        if not isinstance(offset, int):
            raise ValueError("The offset must be an integer")

        self.value = value
        self.offset = offset

    def __call__(self, axis):
        return axis.index(self.value) + self.offset

    def __add__(self, offset):
        return self.__class__(self.value, self.offset + offset)

    def __sub__(self, offset):
        return self.__class__(self.value, self.offset - offset)

    def __repr__(self):
        s = "{self.__class__.__name__}({self.value}"
        if offset != 0:
            s += ", offset={self.offset}"
        s += ")"
        return s.format(self=self)


class rebin(object):
    __slots__ = ("factor",)
    projection = False

    def __init__(self, value):
        self.factor = value

    def __repr__(self):
        return "{self.__class__.__name__}({self.factor})".format(self=self)


class project(object):
    projection = True

    def __new__(cls, *args, **kargs):
        return sum(*args, **kargs)


def underflow(axis):
    return -1


def overflow(axis):
    return len(axis)


def end(axis):
    return len(axis) - 1


class at(object):
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __call__(self, axis):
        if self.value < -2:
            raise IndexError("Index cannot be less than -1")

        return self.value
