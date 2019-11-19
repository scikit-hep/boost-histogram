from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

__all__ = (
    "AxisTransform",
    "Pow",
    "Function",
    "Numbafy",
    "PythonFunction",
    "sqrt",
    "log",
    "convert",
)

from ..._internal.axis_transform import AxisTransform, Pow, Function
from ..._internal import axis_transform_convert as _convert
from ..._core.axis import transform as _atc
from . import convert
from ..._internal.utils import register

sqrt = Pow(0.5)
log = Function("_log_fn", "_exp_fn", convert=_convert.internal_conversion, name="log")


@register()
class Numbafy(Function):
    def __new__(cls, forward, inverse, **kwargs):
        """
        Automatically run numba on the functions given. Requires numba to be installed.

        Identical to Function(forward, inverse, convert=bh.axis.convert.numba)
        """

        return Function(forward, inverse, convert=convert.numba, **kwargs)


@register()
class PythonFunction(Function):
    def __new__(cls, forward, inverse, **kwargs):
        """
        Allow pure python functions to be used in a transform. Will be slower than a compiled function!

        Identical to Function(forward, inverse, convert=bh.axis.convert.python)
        """

        return Function(forward, inverse, convert=convert.python, **kwargs)


del register
