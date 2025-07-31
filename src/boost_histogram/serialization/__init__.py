from __future__ import annotations

from typing import Any

# pylint: disable-next=import-error
from .. import histogram, version
from ._axis import _axis_from_dict, _axis_to_dict
from ._storage import _data_from_dict, _storage_from_dict, _storage_to_dict

__all__ = ["from_uhi", "to_uhi"]


def __dir__() -> list[str]:
    return __all__


def to_uhi(h: histogram.Histogram, /) -> dict[str, Any]:
    """Convert an Histogram to a dictionary."""

    # Convert the histogram to a dictionary
    data = {
        "writer_info": {"boost-histogram": {"version": version.version}, "uhi": 1},
        "axes": [_axis_to_dict(axis) for axis in h.axes],
        "storage": _storage_to_dict(h.storage_type(), h.view(flow=True)),
    }
    if h.metadata is not None:
        data["metadata"] = h.metadata

    return data


def from_uhi(data: dict[str, Any], /) -> histogram.Histogram:
    """Convert a dictionary to an Histogram."""

    h = histogram.Histogram(
        *(_axis_from_dict(ax) for ax in data["axes"]),
        storage=_storage_from_dict(data["storage"]),
        metadata=data.get("metadata"),
    )
    h[...] = _data_from_dict(data["storage"])
    return h
