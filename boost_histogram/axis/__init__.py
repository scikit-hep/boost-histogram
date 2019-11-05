from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from .._internal.axis import Axis, options
from .._internal.axis import Regular, Variable, Integer, Category

from . import transform


# Workarounds for smooth transitions from 0.5 series. Will be removed in later release.
regular = Regular
variable = Variable
integer = Integer
category = Category


def regular_log(*args, **kargs):
    import warnings

    warnings.warn("Use transform=axis.transform.Log instead", DeprecationWarning)
    return Regular(*args, **kargs, transform=transform.Log)


def regular_sqrt(*args, **kargs):
    import warnings

    warnings.warn("Use transform=axis.transform.Sqrt instead", DeprecationWarning)
    return Regular(*args, **kargs, transform=transform.Sqrt)


def regular_pow(bins, start, stop, power, **kargs):
    import warnings

    warnings.warn(
        "Use transform=axis.transform.Pow({0}) instead".format(power),
        DeprecationWarning,
    )
    return Regular(bins, start, stop, **kargs, transform=transform.Pow(power))


def circular(*args, **kargs):
    import warnings

    warnings.warn("Use circular=True instead", DeprecationWarning)
    return Regular(*args, **kargs, circular=True)
