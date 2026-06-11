from __future__ import annotations

import copy
from typing import Any, TypeVar

import numpy as np

# pylint: disable-next=import-error
from .. import histogram, storage, version
from ._axis import _axis_from_dict, _axis_to_dict
from ._common import serialize_metadata
from ._storage import (
    _data_from_dict,
    _storage_from_dict,
    _storage_to_dict,
    _storage_type_to_str,
)

__all__ = ["from_uhi", "remove_writer_info", "to_uhi"]


def __dir__() -> list[str]:
    return __all__


def _storage_has_data_keys(storage_data: dict[str, Any], storage_type: str) -> bool:
    """
    Check if storage data dict has the required keys for the given storage type.

    Returns True if all required data keys are present, False if it's structure-only.
    Raises ValueError if required keys are missing (malformed/partial data).
    """
    match storage_type:
        case "int" | "double":
            required_keys = {"values"}
        case "weighted":
            required_keys = {"values", "variances"}
        case "mean":
            required_keys = {"counts", "values", "variances"}
        case "weighted_mean":
            required_keys = {
                "sum_of_weights",
                "sum_of_weights_squared",
                "values",
                "variances",
            }
        case _:
            msg = f"Unknown storage type: {storage_type}"
            raise ValueError(msg)

    present_keys = required_keys & set(storage_data.keys())

    if not present_keys:
        return False

    if present_keys != required_keys:
        missing = required_keys - present_keys
        msg = f"{storage_type.capitalize()} storage missing required keys: {missing}"
        raise ValueError(msg)

    return True


def to_uhi(
    h: histogram.Histogram[Any], /, *, keep_storage: bool = True
) -> dict[str, Any]:
    """Convert an Histogram to a dictionary."""

    # Convert the histogram to a dictionary
    writer_info = {"boost-histogram": {"version": version.version}}

    # The storage_type property does a subclass-walking cast on every access,
    # so look it up once; a single instance (which for MultiCell carries the
    # real nelem) serves for dispatch and error messages.
    storage_type = h.storage_type
    storage_obj = h.storage

    # Store storage type info for AtomicInt64 and Unlimited (they serialize as int/double)
    storage_type_str = _storage_type_to_str(storage_obj)
    if issubclass(storage_type, (storage.AtomicInt64, storage.Unlimited)):
        writer_info["boost-histogram"]["storage_type"] = storage_type.__name__

    data = {
        "uhi_schema": 1,
        "writer_info": writer_info,
        "axes": [_axis_to_dict(axis) for axis in h.axes],
    }
    if keep_storage:
        data["storage"] = _storage_to_dict(storage_obj, h.view(flow=True))
    else:
        data["storage"] = {"type": storage_type_str}
    data["metadata"] = serialize_metadata(h.__dict__)

    return data


def from_uhi(data: dict[str, Any], /) -> histogram.Histogram[Any]:
    """Convert a dictionary to an Histogram."""
    # One time use
    axis = (_axis_from_dict(ax) for ax in data["axes"])

    storage_data = data["storage"]
    storage_ = _storage_from_dict(storage_data, data.get("writer_info", {}))
    h = histogram.Histogram[Any](*axis, storage=storage_)
    h.__dict__ = data.get("metadata", {})

    # Check if storage has data (if not, it's a structure-only histogram)
    # Validate required keys per storage type before deciding to skip data loading
    storage_type = storage_data["type"]
    has_data_keys = _storage_has_data_keys(storage_data, storage_type)

    if not has_data_keys:
        return h

    raw_data = _data_from_dict(storage_data)
    view_shape = h.view(flow=True).shape
    # Reshape raw_data to the expected shape. This is necessary because JSON
    # serialization can collapse empty dimensions (e.g. (5, 0, 0) -> (5, 0)),
    # so we must restore the correct number of dimensions.
    if storage_type in {"weighted", "mean", "weighted_mean"}:
        raw_data = np.asarray(raw_data)
        raw_data = raw_data.reshape(view_shape + raw_data.shape[-1:])
    else:
        raw_data = np.reshape(raw_data, view_shape)
    h[...] = raw_data
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
        obj.pop("writer_info", None)
    elif library in obj.get("writer_info", {}):
        obj["writer_info"] = copy.copy(obj["writer_info"])
        del obj["writer_info"][library]

    if "axes" in obj:
        obj["axes"] = [remove_writer_info(ax, library=library) for ax in obj["axes"]]
    if "storage" in obj:
        obj["storage"] = remove_writer_info(obj["storage"], library=library)

    return obj
