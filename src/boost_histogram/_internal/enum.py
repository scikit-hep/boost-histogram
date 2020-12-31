# -*- coding: utf-8 -*-
# Try to import Enum, but if it fails, not worth breaking over.

try:
    from enum import Enum
except ImportError:
    try:
        from enum34 import Enum  # type: ignore
    except ImportError:
        Enum = object  # type: ignore


# This is a StrEnum as defined in Python 3.10
class Kind(str, Enum):
    COUNT = "COUNT"
    MEAN = "MEAN"

    __str__ = str.__str__  # type: ignore
