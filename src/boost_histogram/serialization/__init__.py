from __future__ import annotations

import copy
from typing import Any, TypeVar

import numpy as np

# pylint: disable-next=import-error
from .. import histogram, version
from ._axis import _axis_from_dict, _axis_to_dict
from ._common import serialize_metadata
from ._storage import _data_from_dict, _storage_from_dict, _storage_to_dict

__all__ = ["from_uhi", "remove_writer_info", "to_uhi"]


def __dir__() -> list[str]:
    return __all__


def to_uhi(
    h: histogram.Histogram[Any], /, *, keep_storage: bool = True
) -> dict[str, Any]:
    """Convert an Histogram to a dictionary."""

    # Convert the histogram to a dictionary
    data = {
        "uhi_schema": 1,
        "writer_info": {"boost-histogram": {"version": version.version}},
        "axes": [_axis_to_dict(axis) for axis in h.axes],
    }
    if keep_storage:
        data["storage"] = _storage_to_dict(h.storage_type(), h.view(flow=True))
    data["metadata"] = serialize_metadata(h.__dict__)

    return data


def from_uhi(data: dict[str, Any], /) -> histogram.Histogram[Any]:
    """Convert a dictionary to an Histogram."""
    # One time use
    axis = (_axis_from_dict(ax) for ax in data["axes"])

    if "storage" not in data:
        h = histogram.Histogram[Any](*axis)
        h.__dict__ = data.get("metadata", {})
        return h

    storage = _storage_from_dict(data["storage"])
    h = histogram.Histogram[Any](*axis, storage=storage)

    raw_data = _data_from_dict(data["storage"])
    view_shape = h.view(flow=True).shape
    # Reshape raw_data to the expected shape. This is necessary because JSON
    # serialization can collapse empty dimensions (e.g. (5, 0, 0) -> (5, 0)),
    # so we must restore the correct number of dimensions.
    storage_type = data["storage"]["type"]
    if storage_type in {"weighted", "mean", "weighted_mean"}:
        raw_data = np.asarray(raw_data)
        raw_data = raw_data.reshape(view_shape + raw_data.shape[-1:])
    else:
        raw_data = np.reshape(raw_data, view_shape)
    h[...] = raw_data
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
