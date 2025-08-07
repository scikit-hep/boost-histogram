from __future__ import annotations

import copy
from typing import Any, TypeVar

# pylint: disable-next=import-error
from .. import histogram, version
from ._axis import _axis_from_dict, _axis_to_dict
from ._storage import _data_from_dict, _storage_from_dict, _storage_to_dict

__all__ = ["from_uhi", "remove_writer_info", "to_uhi"]


def __dir__() -> list[str]:
    return __all__


def to_uhi(h: histogram.Histogram, /) -> dict[str, Any]:
    """Convert an Histogram to a dictionary."""

    # Convert the histogram to a dictionary
    data = {
        "uhi_schema": 1,
        "writer_info": {"boost-histogram": {"version": version.version}},
        "axes": [_axis_to_dict(axis) for axis in h.axes],
        "storage": _storage_to_dict(h.storage_type(), h.view(flow=True)),
    }
    if h.metadata is not None:
        data["metadata"] = {
            k: v for k, v in h.metadata.items() if not k.startswith("@")
        }

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


T = TypeVar("T", bound="dict[str, Any]")


def remove_writer_info(obj: T) -> T:
    """Removes all boost-histogram writer_info from a histogram dict, axes dict, or storage dict. Makes copies where required, and the outer dictionary is always copied."""

    obj = copy.copy(obj)
    if "boost-histogram" in obj.get("writer_info", {}):
        obj["writer_info"] = copy.copy(obj["writer_info"])
        del obj["writer_info"]["boost-histogram"]

    if "axes" in obj:
        obj["axes"] = [remove_writer_info(ax) for ax in obj["axes"]]
    if "storage" in obj:
        obj["storage"] = remove_writer_info(obj["storage"])

    return obj
