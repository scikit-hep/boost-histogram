from __future__ import annotations

import sys
import typing

if sys.version_info >= (3, 11):
    from typing import Self
elif typing.TYPE_CHECKING:
    from typing_extensions import Self
else:
    Self = object

if sys.version_info >= (3, 12):
    from typing import TypeVar
elif typing.TYPE_CHECKING:
    from typing_extensions import TypeVar
else:

    def TypeVar(*args, default=None, **kwargs):  # noqa: ARG001
        return typing.TypeVar(*args, **kwargs)


__all__ = ["Self", "TypeVar"]


def __dir__() -> list[str]:
    return __all__
