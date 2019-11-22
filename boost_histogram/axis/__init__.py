from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

__all__ = ("Regular", "Variable", "Integer", "Category", "Axis", "options", "transform")

from .._internal.axis import Axis, options
from .._internal.utils import register
from .._internal.axis import (
    Regular,
    Variable,
    Integer,
    Category,
    IntCategory,
    StrCategory,
)
from . import transform

import warnings as _warnings

# Workarounds for smooth transitions from 0.5 series. Will be removed in later release.

# Normally, further-down classes will be registered and used,
# But we don't want these used.
@register()
class regular(Regular):
    def __init__(self, *args, **kwargs):
        _warnings.warn("Use Regular instead", FutureWarning)
        return super(regular, self).__init__(*args, **kwargs)


@register()
class variable(Variable):
    _CLASSES = set()

    def __init__(self, *args, **kwargs):
        _warnings.warn("Use Variable instead", FutureWarning)
        return super(variable, self).__init__(*args, **kwargs)


@register()
class integer(Integer):
    _CLASSES = set()

    def __init__(self, *args, **kwargs):
        _warnings.warn("Use Integer instead", FutureWarning)
        return super(integer, self).__init__(*args, **kwargs)


def category(*args, **kwargs):
    _warnings.warn("Use IntCategory or StrCategory instead", FutureWarning)
    return Category(*args, **kwargs)


def regular_log(*args, **kwargs):
    _warnings.warn("Use transform=axis.transform.log instead", FutureWarning)
    return Regular(*args, transform=transform.log, **kwargs)


def regular_sqrt(*args, **kwargs):
    _warnings.warn("Use transform=axis.transform.sqrt instead", FutureWarning)
    return Regular(*args, transform=transform.sqrt, **kwargs)


def regular_pow(bins, start, stop, power, **kwargs):
    _warnings.warn(
        "Use transform=axis.transform.Pow({0}) instead".format(power), FutureWarning
    )
    return Regular(bins, start, stop, transform=transform.Pow(power), **kwargs)


def circular(*args, **kwargs):
    _warnings.warn("Use circular=True instead", FutureWarning)
    return Regular(*args, circular=True, **kwargs)


del register
