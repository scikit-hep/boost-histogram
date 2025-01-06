from __future__ import annotations

from .._internal.axis_transform import (
    AxisTransform,
    Function,
    Pow,
    _internal_conversion,
)

__all__ = ("AxisTransform", "Function", "Pow", "log", "sqrt")

sqrt = Function("_sqrt_fn", "_sq_fn", convert=_internal_conversion, name="sqrt")
log = Function("_log_fn", "_exp_fn", convert=_internal_conversion, name="log")
