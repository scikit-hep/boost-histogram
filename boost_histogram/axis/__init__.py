from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

__all__ = ("Regular", "Variable", "Integer", "Category", "Axis", "options", "transform")

from .._internal.axis import Axis, options
from .._internal.utils import register as _register
from .._internal.axis import Regular, Variable, Integer, Category
from . import transform

import warnings as _warnings

# Workarounds for smooth transitions from 0.5 series. Will be removed in later release.

# Normally, further-down classes will be registered and used,
# But we don't want these used.
@_register()
class regular(Regular):
    def __init__(self, *args, **kwargs):
        _warnings.warn("Use Regular instead", DeprecationWarning)
        return super(regular, self).__init__(*args, **kwargs)


@_register()
class variable(Variable):
    _CLASSES = set()

    def __init__(self, *args, **kwargs):
        _warnings.warn("Use Variable instead", DeprecationWarning)
        return super(variable, self).__init__(*args, **kwargs)


@_register()
class integer(Integer):
    _CLASSES = set()

    def __init__(self, *args, **kwargs):
        _warnings.warn("Use Integer instead", DeprecationWarning)
        return super(integer, self).__init__(*args, **kwargs)


@_register()
class category(Category):
    _CLASSES = set()

    def __init__(self, *args, **kwargs):
        _warnings.warn("Use Category instead", DeprecationWarning)
        return super(category, self).__init__(*args, **kwargs)


def regular_log(*args, **kwargs):
    _warnings.warn("Use transform=axis.transform.Log() instead", DeprecationWarning)
    return Regular(*args, transform=Transform.Log, **kwargs)


def regular_sqrt(*args, **kwargs):
    _warnings.warn("Use transform=axis.transform.Sqrt() instead", DeprecationWarning)
    return Regular(*args, transform=transform.Sqrt, **kwargs)


def regular_pow(bins, start, stop, power, **kwargs):
    _warnings.warn(
        "Use transform=axis.transform.Pow({0}) instead".format(power),
        DeprecationWarning,
    )
    return Regular(bins, start, stop, transform=transform.Pow(power), **kwargs)


def circular(*args, **kwargs):
    _warnings.warn("Use circular=True instead", DeprecationWarning)
    return Regular(*args, circular=True, **kwargs)
