from __future__ import annotations

from typing import Any


def serialize_metadata(value: dict[str, Any], /) -> dict[str, Any]:
    return {k: v for k, v in value.items() if not k.startswith("@") and v is not None}
