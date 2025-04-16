from __future__ import annotations

from typing import Any

from .. import Histogram
from ._axis import _axes_from_dict, _axis_to_dict
from ._storage import _data_from_dict, _storage_from_dict, _storage_to_dict

__all__ = ["from_dict", "to_dict"]


def __dir__() -> list[str]:
    return __all__


def to_dict(h: Histogram, /) -> dict[str, Any]:
    """Convert an Histogram to a dictionary."""

    # Convert the histogram to a dictionary
    data = {
        "axes": [_axis_to_dict(axis) for axis in h.axes],
        "storage": _storage_to_dict(h.storage_type(), h.view(flow=True)),
    }
    if h.metadata is not None:
        data["metadata"] = h.metadata

    return data


def from_dict(data: dict[str, Any], /) -> Histogram:
    """Convert a dictionary to an Histogram."""

    h = Histogram(
        *_axes_from_dict(data["axes"]),
        storage=_storage_from_dict(data["storage"]),
        metadata=data.get("metadata"),
    )
    h[...] = _data_from_dict(data["storage"])
    return h
