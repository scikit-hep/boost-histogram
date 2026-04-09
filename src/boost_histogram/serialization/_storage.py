from __future__ import annotations

import functools
from typing import Any

import numpy as np

from .. import storage

__all__ = [
    "_data_from_dict",
    "_storage_from_dict",
    "_storage_to_dict",
    "_storage_type_to_str",
]


def __dir__() -> list[str]:
    return __all__


def _storage_type_to_str(_storage: storage.Storage, /) -> str:
    """Return the canonical storage type string for a storage object."""
    match _storage:
        case storage.Int64():
            return "int"
        case storage.Double():
            return "double"
        case storage.AtomicInt64():
            return "int"
        case storage.Unlimited():
            return "double"
        case storage.Weight():
            return "weighted"
        case storage.Mean():
            return "mean"
        case storage.WeightedMean():
            return "weighted_mean"
        case _:
            raise TypeError(f"Unsupported storage type: {_storage}")


@functools.singledispatch
def _storage_to_dict(_storage: storage.Storage, /, data: Any) -> dict[str, Any]:  # noqa: ARG001
    """Convert a storage to a dictionary."""
    msg = f"Unsupported storage type: {_storage}"
    raise TypeError(msg)


@_storage_to_dict.register(storage.Int64)
@_storage_to_dict.register(storage.Double)
def _(_storage: storage.Int64 | storage.Double, /, data: Any) -> dict[str, Any]:
    return {"type": _storage_type_to_str(_storage), "values": data}


@_storage_to_dict.register(storage.AtomicInt64)
@_storage_to_dict.register(storage.Unlimited)
def _(
    _storage: storage.AtomicInt64 | storage.Unlimited,
    /,
    data: Any,
) -> dict[str, Any]:
    return {
        "type": "int" if np.issubdtype(data.dtype, np.integer) else "double",
        "values": data,
    }


@_storage_to_dict.register(storage.Weight)
def _(_storage: storage.Weight, /, data: Any) -> dict[str, Any]:
    return {
        "type": _storage_type_to_str(_storage),
        "values": data.value,
        "variances": data.variance,
    }


@_storage_to_dict.register(storage.Mean)
def _(_storage: storage.Mean, /, data: Any) -> dict[str, Any]:
    return {
        "type": _storage_type_to_str(_storage),
        "counts": data.count,
        "values": data.value,
        "variances": data.variance,
    }


@_storage_to_dict.register(storage.WeightedMean)
def _(_storage: storage.WeightedMean, /, data: Any) -> dict[str, Any]:
    return {
        "type": _storage_type_to_str(_storage),
        "sum_of_weights": data.sum_of_weights,
        "sum_of_weights_squared": data.sum_of_weights_squared,
        "values": data.value,
        "variances": data.variance,
    }


def _storage_from_dict(
    data: dict[str, Any], writer_info: dict[str, Any] | None = None, /
) -> storage.Storage:
    """Convert a dictionary to a storage object."""
    # If loading a boost-histogram, we can load the exact original type
    # Check both the main writer_info (new location) and storage writer_info (old location)
    orig_type = ""
    if writer_info:
        orig_type = writer_info.get("boost-histogram", {}).get("storage_type", "")
    if not orig_type:
        orig_type = (
            data.get("writer_info", {}).get("boost-histogram", {}).get("orig_type", "")
        )

    if orig_type == "AtomicInt64":
        return storage.AtomicInt64()
    if orig_type == "Unlimited":
        return storage.Unlimited()

    storage_type = data["type"]
    if storage_type == "int":
        return storage.Int64()
    if storage_type == "double":
        return storage.Double()
    if storage_type == "weighted":
        return storage.Weight()
    if storage_type == "mean":
        return storage.Mean()
    if storage_type == "weighted_mean":
        return storage.WeightedMean()

    raise TypeError(f"Unsupported storage type: {storage_type}")


def _data_from_dict(data: dict[str, Any], /) -> np.typing.NDArray[Any]:
    """Convert a dictionary to data."""
    storage_type = data["type"]

    if storage_type in {"int", "double"}:
        return data["values"]  # type: ignore[no-any-return]
    if storage_type == "weighted":
        return np.stack([data["values"], data["variances"]], axis=-1)
    if storage_type == "mean":
        return np.stack([data["counts"], data["values"], data["variances"]], axis=-1)
    if storage_type == "weighted_mean":
        return np.stack(
            [
                data["sum_of_weights"],
                data["sum_of_weights_squared"],
                data["values"],
                data["variances"],
            ],
            axis=-1,
        )

    raise TypeError(f"Unsupported storage type: {storage_type}")
