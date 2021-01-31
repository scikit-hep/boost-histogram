# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import copy
from typing import Any

from .._core import axis as ca
from .kwargs import KWArgs
from .sig_tools import inject_signature
from .utils import MAIN_FAMILY, register, set_family, set_module

del absolute_import, division, print_function


@set_module("boost_histogram.axis.transform")
class AxisTransform(object):
    __slots__ = ("_this",)

    def __copy__(self):
        other = self.__class__.__new__(self.__class__)
        other._this = copy.copy(self._this)
        return other

    @classmethod
    def _convert_cpp(cls, this):
        self = cls.__new__(cls)
        self._this = this
        return self

    def __repr__(self):
        if hasattr(self, "_this"):
            return repr(self._this)
        else:
            return self.__class__.__name__ + "() # Missing _this, broken class"

    def _produce(self, bins, start, stop):
        # type: (int, int, int) -> Any
        # Note: this is an ABC; _type must be defined on children
        # These can be fixed later with a Protocol
        return self.__class__._type(bins, start, stop)  # type: ignore

    def __init__(self):
        # type: () -> None
        "Create a new transform instance"
        # Note: this comes from set_family
        (cpp_class,) = self._types  # type: ignore
        self._this = cpp_class()

    def forward(self, value):
        "Compute the forward transform"
        return self._this.forward(value)

    def inverse(self, value):
        "Compute the inverse transform"
        return self._this.inverse(value)


core = "__init__ forward inverse".split()


@set_family(MAIN_FAMILY)
@set_module("boost_histogram.axis.transform")
@register({ca.transform.pow})
class Pow(AxisTransform):
    __slots__ = ()
    _type = ca.regular_pow

    def __init__(self, power):
        "Create a new transform instance"
        # Note: this comes from set_family
        (cpp_class,) = self._types  # type: ignore
        self._this = cpp_class(power)

    @property
    def power(self):
        "The power of the transform"
        return self._this.power

    # This one does need to be a normal method
    def _produce(self, bins, start, stop):
        return self.__class__._type(bins, start, stop, self.power)


@set_family(MAIN_FAMILY)
@set_module("boost_histogram.axis.transform")
@register({ca.transform.func_transform})
class Function(AxisTransform):
    __slots__ = ()
    _type = ca.regular_trans

    @inject_signature("self, forward, inverse, *, convert=None, name=''")
    def __init__(self, forward, inverse, **kwargs):
        """
        Create a functional transform from a ctypes double(double) function
        pointer or any object that provides such an interface through a
        ``.ctypes`` attribute (such as numba.cfunc). A pure python function *can*
        be adapted to a ctypes pointer, but please use a Variable axis instead or
        use something like numba to produce a compiled function pointer. You can
        manually specify the repr name with ``name=``.

        Example of Numba use:
        ---------------------

            @numba.cfunc(numba.float64(numba.float64,))
            def exp(x):
                return math.exp(x)

            @numba.cfunc(numba.float64(numba.float64,))
            def log(x):
                return math.log(x)

        Example of slow CTypes use:
        ---------------------------

            ftype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
            log = ftype(math.log)
            exp = ftype(math.exp)


        Now you can supply these functions, and you will get a high performance
        transformation axis.

        You can also supply an optional conversion function; this will take the input
        forward and inverse and call them before producing a transform. This enables
        pickling, as well, since ctypes pointers are not picklable. A few common
        utilities have been supplied:

        * ``convert.numba``: Compile using numba (required)
        * ``convert.python``: Just call the Python function (15-90x slower than compiled)

        See also
        --------

        * ``Numbify(forward, inverse, *, name='')``: Uses convert=convert.numba
        * ``PythonFunction(forward, inverse, *, name='')``: Uses convert=convert.python

        """

        with KWArgs(kwargs) as k:
            convert = k.optional("convert")
            name = k.optional("name", "")

        # Note: this comes from set_family
        (cpp_class,) = self._types  # type: ignore
        self._this = cpp_class(forward, inverse, convert, name)

    # This one does need to be a normal method
    def _produce(self, bins, start, stop):
        return self.__class__._type(bins, start, stop, self._this)


def _internal_conversion(value):
    return getattr(ca.transform, value)
