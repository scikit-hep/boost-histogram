from __future__ import annotations

import sys
import typing

if sys.version_info >= (3, 11):
    from typing import Self
elif typing.TYPE_CHECKING:
    from typing_extensions import Self
else:
    Self = object

__all__ = ["Self"]


def __dir__() -> list[str]:
    return __all__
