class FactoryMeta(object):
    def __init__(self, f, types):
        self._f = f
        self._types = types

    def __call__(self, *args, **kwargs):
        return self._f(*args, **kwargs)

    def __instancecheck__(self, other):
        return isinstance(other, self._types)
