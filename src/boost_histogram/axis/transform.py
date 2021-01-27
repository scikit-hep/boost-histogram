# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from .._internal.axis_transform import (
    AxisTransform,
    Function,
    Pow,
    _internal_conversion,
)

del absolute_import, division, print_function

__all__ = ("AxisTransform", "Pow", "Function", "sqrt", "log")

sqrt = Function("_sqrt_fn", "_sq_fn", convert=_internal_conversion, name="sqrt")
log = Function("_log_fn", "_exp_fn", convert=_internal_conversion, name="log")
