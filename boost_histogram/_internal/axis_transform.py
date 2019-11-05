from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from .._core import axis as ca


class AxisTransform(object):
    __slots__ = ()


class Log(AxisTransform):
    @staticmethod
    def _produce(bins, start, stop, metadata):
        return ca.regular_log(bins, start, stop, metadata)


class Sqrt(AxisTransform):
    @staticmethod
    def _produce(bins, start, stop, metadata):
        return ca.regular_sqrt(bins, start, stop, metadata)


class Pow(AxisTransform):
    __slots__ = ("power",)

    def __init__(self, power):
        self.power = power

    def _produce(self, bins, start, stop, metadata):
        return ca.regular_pow(bins, start, stop, self.power, metadata)
