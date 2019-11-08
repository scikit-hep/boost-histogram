from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

__all__ = ("options", "regular", "variable", "integer", "category", "transorm")

from ...axis import (
    options,
    Regular as regular,
    Variable as variable,
    Integer as integer,
    Category as category,
)

# TODO: Make these proper classes with proper names/module

from . import transform
