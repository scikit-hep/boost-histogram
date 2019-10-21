from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function


class loc(object):
    __slots__ = ("value", "offset")

    def __init__(self, value, offset=0):
        if not isinstance(offset, int):
            raise ValueError("The offset must be an integer")

        self.value = value
        self.offset = offset

    def __add__(self, offset):
        return self.__class__(self.value, self.offset + offset)

    def __sub__(self, offset):
        return self.__class__(self.value, self.offset - offset)


class rebin(object):
    __slots__ = ("factor",)
    projection = False

    def __init__(self, value):
        self.factor = value


class project(object):
    projection = True


class underflow(object):
    flow = -1


class overflow(object):
    flow = 1
