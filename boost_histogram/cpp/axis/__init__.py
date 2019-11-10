from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

__all__ = ("options", "regular", "variable", "integer", "category", "transorm")

from ... import axis as _axis
from ..._internal.utils import cpp_module as _cpp_module

from ...axis import options
from . import transform


@_cpp_module
class regular(_axis.Regular):
    def __repr__(self):
        return repr(self._ax)


@_cpp_module
class variable(_axis.Variable):
    def __repr__(self):
        return repr(self._ax)


@_cpp_module
class integer(_axis.Integer):
    def __repr__(self):
        return repr(self._ax)


@_cpp_module
class category(_axis.Category):
    def __repr__(self):
        return repr(self._ax)
