from __future__ import annotations

import functools
from typing import Any

import numpy as np

from .. import storage

__all__ = ["_data_from_dict", "_storage_from_dict", "_storage_to_dict"]


def __dir__() -> list[str]:
    return __all__


@functools.singledispatch
def _storage_to_dict(_storage: Any, /, data: Any) -> dict[str, Any]:  # noqa: ARG001
    """Convert a storage to a dictionary."""
    msg = f"Unsupported storage type: {_storage}"
    raise TypeError(msg)


@_storage_to_dict.register(storage.Int64)
def _(_storage: storage.Int64, /, data: Any) -> dict[str, Any]:
    return {"type": "int", "values": data}


@_storage_to_dict.register(storage.Double)
def _(_storage: storage.Double, /, data: Any) -> dict[str, Any]:
    return {"type": "double", "values": data}


@_storage_to_dict.register(storage.AtomicInt64)
@_storage_to_dict.register(storage.Unlimited)
def _(
    storage_: storage.AtomicInt64 | storage.Unlimited,
    /,
    data: Any,
) -> dict[str, Any]:
    return {
        "writer_info": {"boost-histogram": {"orig_type": type(storage_).__name__}},
        "type": "int" if np.issubdtype(data.dtype, np.integer) else "double",
        "values": data,
    }


@_storage_to_dict.register(storage.Weight)
def _(_storage: storage.Weight, /, data: Any) -> dict[str, Any]:
    return {
        "type": "weighted",
        "values": data.value,
        "variances": data.variance,
    }


@_storage_to_dict.register(storage.Mean)
def _(_storage: storage.Mean, /, data: Any) -> dict[str, Any]:
    return {
        "type": "mean",
        "counts": data.count,
        "values": data.value,
        "variances": data.variance,
    }


@_storage_to_dict.register(storage.WeightedMean)
def _(_storage: storage.WeightedMean, /, data: Any) -> dict[str, Any]:
    return {
        "type": "weighted_mean",
        "sum_of_weights": data.sum_of_weights,
        "sum_of_weights_squared": data.sum_of_weights_squared,
        "values": data.value,
        "variances": data.variance,
    }


def _storage_from_dict(data: dict[str, Any], /) -> storage.Storage:
    """Convert a dictionary to a storage object."""
    # If loading a boost-histogram, we can load the exact original type
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
        return data["values"]
    if storage_type == "weighted":
        return np.stack([data["values"], data["variances"]]).T
    if storage_type == "mean":
        return np.stack(
            [data["counts"], data["values"], data["variances"]],
        ).T
    if storage_type == "weighted_mean":
        return np.stack(
            [
                data["sum_of_weights"],
                data["sum_of_weights_squared"],
                data["values"],
                data["variances"],
            ],
        ).T

    raise TypeError(f"Unsupported storage type: {storage_type}")
