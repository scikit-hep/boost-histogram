from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

from ._internal.axis import Axis, options
from ._internal.axis import (
    Regular as regular,
    Variable as variable,
    Integer as integer,
    Category as category,
)

circular = regular.circular
regular_log = regular.log
regular_pow = regular.pow
regular_sqrt = regular.sqrt
