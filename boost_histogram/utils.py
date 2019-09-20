class FactoryMeta(object):
    def __init__(self, f, types):
        self._f = f
        self._types = types

    def __call__(self, *args, **kwargs):
        return self._f(*args, **kwargs)

    def __instancecheck__(self, other):
        return isinstance(other, self._types)


class loc(object):
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


class rebin(object):
    __slots__ = ("factor",)
    projection = False

    def __init__(self, value):
        self.factor = value


class _project(object):
    __slots__ = ()
    projection = True


project = _project


def indexed(histogram, flow=False):
    """Set up an iterator, returns a special accessor for bin info and content."""
    return histogram.indexed(flow=flow)
