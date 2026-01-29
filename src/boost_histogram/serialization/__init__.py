from __future__ import annotations

import copy
from typing import Any, TypeVar

# pylint: disable-next=import-error
from .. import histogram, version
from ._axis import _axis_from_dict, _axis_to_dict
from ._common import serialize_metadata
from ._storage import _data_from_dict, _storage_from_dict, _storage_to_dict

__all__ = ["from_uhi", "remove_writer_info", "to_uhi"]


def __dir__() -> list[str]:
    return __all__


def to_uhi(h: histogram.Histogram[Any], /) -> dict[str, Any]:
    """Convert an Histogram to a dictionary."""

    # Convert the histogram to a dictionary
    data = {
        "uhi_schema": 1,
        "writer_info": {"boost-histogram": {"version": version.version}},
        "axes": [_axis_to_dict(axis) for axis in h.axes],
        "storage": _storage_to_dict(h.storage_type(), h.view(flow=True)),
    }
    data["metadata"] = serialize_metadata(h.__dict__)

    return data


def from_uhi(data: dict[str, Any], /) -> histogram.Histogram[Any]:
    """Convert a dictionary to an Histogram."""

    h = histogram.Histogram(
        *(_axis_from_dict(ax) for ax in data["axes"]),
        storage=_storage_from_dict(data["storage"]),
    )
    h[...] = _data_from_dict(data["storage"])
    h.__dict__ = data.get("metadata", {})
    return h


T = TypeVar("T", bound="dict[str, Any]")


def remove_writer_info(obj: T, /, *, library: str | None = "boost-histogram") -> T:
    """
    Removes all ``writer_info`` for a library from a histogram dict, axes dict,
    or storage dict. Makes copies where required, and the outer dictionary is
    always copied.

    Specify a library name, or ``None`` to remove all.
    """

    obj = copy.copy(obj)
    if library is None:
        obj.pop("writer_info")
    elif library in obj.get("writer_info", {}):
        obj["writer_info"] = copy.copy(obj["writer_info"])
        del obj["writer_info"][library]

    if "axes" in obj:
        obj["axes"] = [remove_writer_info(ax, library=library) for ax in obj["axes"]]
    if "storage" in obj:
        obj["storage"] = remove_writer_info(obj["storage"], library=library)

    return obj
