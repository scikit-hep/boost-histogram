from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from .._core import axis as ca

from .utils import register


class AxisTransform(object):
    __slots__ = ()

    # This should be a @classmethod because it
    # does not depend on self
    # But if it was, then Func() would be replaceable by Func
    def _produce(self, bins, start, stop, metadata):
        return self.__class__._type(bins, start, stop, metadata)


@register(ca.transform.log)
class Log(ca.transform.log, AxisTransform):
    __slots__ = ()
    _type = ca.regular_log


@register(ca.transform.sqrt)
class Sqrt(ca.transform.sqrt, AxisTransform):
    __slots__ = ()
    _type = ca.regular_sqrt


@register(ca.transform.pow)
class Pow(ca.transform.pow, AxisTransform):
    __slots__ = ()
    _type = ca.regular_pow

    # This one does need to be a normal method
    def _produce(self, bins, start, stop, metadata):
        return self.__class__._type(bins, start, stop, self.power, metadata)
