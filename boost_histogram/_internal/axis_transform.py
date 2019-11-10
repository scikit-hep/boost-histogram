from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from .._core import axis as ca


class AxisTransform(object):
    __slots__ = ()

    # This should be a @classmethod because it
    # does not depend on self
    # But if it was, then Func() would be replaceable by Func
    def _produce(self, bins, start, stop, metadata):
        return self.__class__._type(bins, start, stop, metadata)


class Log(ca.transform.log, AxisTransform):
    __slots__ = ()
    _type = ca.regular_log


class Sqrt(ca.transform.sqrt, AxisTransform):
    __slots__ = ()
    _type = ca.regular_sqrt


class Pow(ca.transform.pow, AxisTransform):
    __slots__ = ()
    _type = ca.regular_pow

    # This one does need to be a normal method
    def _produce(self, bins, start, stop, metadata):
        return self.__class__._type(bins, start, stop, self.power, metadata)


def _to_transform(trans):
    for trans_class in AxisTransform.__subclasses__():
        if isinstance(trans, trans_class.__bases__):
            return trans_class(trans)
    raise TypeError("Cannot find transform")
