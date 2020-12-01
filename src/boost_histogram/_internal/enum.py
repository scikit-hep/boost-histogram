# -*- coding: utf-8 -*-
# Try to import Enum, but if it fails, not worth breaking over.

try:
    from enum import Enum
except ImportError:
    try:
        from enum34 import Enum  # type: ignore
    except ImportError:
        Enum = object  # type: ignore


class Implementation(str, Enum):
    sum = "sum"
    mean = "mean"
