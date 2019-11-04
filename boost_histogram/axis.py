from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from ._internal.axis import Axis, options
from ._internal.axis import (
    Regular as regular,
    Log as regular_log,
    Pow as regular_pow,
    Sqrt as regular_sqrt,
    Variable as variable,
    Integer as integer,
    Category as category,
)

circular = regular.circular
