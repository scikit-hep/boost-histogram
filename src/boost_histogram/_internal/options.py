# -*- coding: utf-8 -*-
# This is basically a dataclass from Python 3.7, with frozen=True

_options = (
    "underflow",
    "overflow",
    "circular",
    "growth",
    "continuous",
    "inclusive",
    "ordered",
)


class Options(object):
    __slots__ = _options

    def __init__(
        self,
        underflow=False,
        overflow=False,
        circular=False,
        growth=False,
        continuous=False,
        inclusive=False,
        ordered=False,
    ):
        for name in _options:
            object.__setattr__(self, name, locals()[name])

    def __eq__(self, other):
        return all(getattr(self, name) == getattr(other, name) for name in _options)

    def __ne__(self, other):
        return not self == other

    @property
    def discrete(self):
        "True if axis is not continuous"
        return not self.continuous

    def __repr__(self):
        args = ("{}={}".format(name, getattr(self, name)) for name in _options)
        return "{}({})".format(self.__class__.__name__, ", ".join(args))
