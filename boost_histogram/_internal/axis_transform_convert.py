from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

import ctypes
import types

from .utils import set_module
from .._core import axis as ca

function_type = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)


@set_module("boost_histogram.axis.transform.convert")
def basic(value):
    if hasattr(value, "ctypes"):
        return value.ctypes
    else:
        return value


def internal_conversion(value):
    return getattr(ca.transform, value)


@set_module("boost_histogram.axis.transform.convert")
def numba(value):
    import numba

    # Support math.sqrt and such
    if isinstance(value, types.BuiltinFunctionType):
        return numba.cfunc(numba.double(numba.double))(lambda x: value(x)).ctypes

    return numba.cfunc(numba.double(numba.double))(value).ctypes


@set_module("boost_histogram.axis.transform.convert")
def python(value):
    return function_type(value)
