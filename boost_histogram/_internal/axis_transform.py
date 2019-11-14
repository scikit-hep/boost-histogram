from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from .._core import axis as ca

from .utils import register, set_family, CPP_FAMILY, MAIN_FAMILY, set_module


@set_module("boost_histogram.axis.transform")
class AxisTransform(object):
    __slots__ = ()

    # This should be a @classmethod because it
    # does not depend on self
    # But if it was, then Func() would be replaceable by Func
    def _produce(self, bins, start, stop, metadata):
        return self.__class__._type(bins, start, stop, metadata)


@set_family(MAIN_FAMILY)
@set_module("boost_histogram.axis.transform")
class Log(ca.transform.log, AxisTransform):
    __slots__ = ()
    _type = ca.regular_log


@set_family(MAIN_FAMILY)
@set_module("boost_histogram.axis.transform")
class Sqrt(ca.transform.sqrt, AxisTransform):
    __slots__ = ()
    _type = ca.regular_sqrt


@set_family(MAIN_FAMILY)
@set_module("boost_histogram.axis.transform")
class Pow(ca.transform.pow, AxisTransform):
    __slots__ = ()
    _type = ca.regular_pow

    # This one does need to be a normal method
    def _produce(self, bins, start, stop, metadata):
        return self.__class__._type(bins, start, stop, self.power, metadata)


### CPP FAMILY ###


@set_family(CPP_FAMILY)
@set_module("boost_histogram.cpp.axis.transform")
class log(Log):
    __slots__ = ()


@set_family(CPP_FAMILY)
@set_module("boost_histogram.cpp.axis.transform")
class sqrt(Sqrt):
    __slots__ = ()


@set_family(CPP_FAMILY)
@set_module("boost_histogram.cpp.axis.transform")
class pow(Pow):
    __slots__ = ()
