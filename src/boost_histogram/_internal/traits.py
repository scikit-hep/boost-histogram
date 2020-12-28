# -*- coding: utf-8 -*-
# This is basically a dataclass from Python 3.7, with frozen=True

_traits = (
    "underflow",
    "overflow",
    "circular",
    "growth",
    "continuous",
    "ordered",
)


# This can be converted to a immutable dataclass once Python < 3.7 is dropped.


class Traits(object):
    __slots__ = _traits

    def __init__(
        self,
        underflow=False,
        overflow=False,
        circular=False,
        growth=False,
        continuous=False,
        ordered=False,
    ):
        for name in _traits:
            setattr(self, name, locals()[name])

    def __eq__(self, other):
        return all(getattr(self, name) == getattr(other, name) for name in _traits)

    def __ne__(self, other):
        return not self == other

    @property
    def discrete(self):
        "True if axis is not continuous"
        return not self.continuous

    def __repr__(self):
        args = ("{}={}".format(name, getattr(self, name)) for name in _traits)
        return "{}({})".format(self.__class__.__name__, ", ".join(args))
