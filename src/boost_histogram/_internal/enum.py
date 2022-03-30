from enum import Enum
from typing import Callable, cast


# This is a StrEnum as defined in Python 3.10
class Kind(str, Enum):
    COUNT = "COUNT"
    MEAN = "MEAN"

    # This cast + type ignore is really odd, so it deserves a quick
    # explanation. If we just set this like StrEnum does, then mypy complains
    # that the type is changing (str -> Kind). If we type: ignore, then
    # MyPy claims that the type: ignore is not needed. If we cast, we get the
    # same error as before. But if we cast and type: ignore, it now works.
    # Will report to MyPy. Tested on 0.800.
    __str__ = cast(Callable[["Kind"], str], str.__str__)  # type: ignore[assignment]
